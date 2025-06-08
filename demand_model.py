import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 确保 Matplotlib 图形字体等不出现乱码（可根据需要调整）
plt.rcParams['font.size'] = 10

# 参数设置
# 基础负荷参数（可根据需要调整数值）
BASE_CONST = 500            # 基础负荷常数项
BASE_SEASON_AMPLITUDE = 100 # 基础负荷年度变化幅度
BASE_DAILY_AMPLITUDE = 50   # 基础负荷日变化幅度

# 温度参数
TEMP_BASE = 20              # 年平均温度（摄氏度）
TEMP_SEASON_AMPLITUDE = 10  # 年度温度变化幅度（夏冬之差的一半）
TEMP_DAILY_AMPLITUDE = 7    # 每日温度日夜变化幅度
TEMP_NOISE_STD = 2          # 温度随机噪声标准差

# 湿度参数
HUMID_BASE = 50             # 年平均相对湿度（百分比）
HUMID_SEASON_AMPLITUDE = 15 # 年度湿度变化幅度
HUMID_DAILY_AMPLITUDE = 15  # 每日湿度日变化幅度
HUMID_NOISE_STD = 5         # 湿度随机噪声标准差

# 占用率参数
# 占用率采用固定模式，无额外随机噪声，以 0~1 表示

# 太阳辐照度参数
SUMMER_MAX_IRR = 1000       # 夏至左右正午峰值辐照度 (假设值, 单位 W/m^2)
WINTER_MIN_IRR = 500        # 冬至左右正午峰值辐照度 (假设值)
DAYLIGHT_VAR_HOURS = 4      # 夏冬日长差异（小时数增减量，±4小时）
IRR_NOISE_FRAC = 0.1        # 辐照度随机扰动幅度（相对百分比）

# 线性模型权重（用于生成实际负荷时的特征贡献权重）
W_TEMP = 10.0    # 温度对负荷的影响系数
W_HUMID = 5.0    # 湿度对负荷的影响系数
W_OCC = 300.0    # 占用率对负荷的影响系数
W_IRR = 0.1      # 辐照度对负荷的影响系数

# 负荷额外噪声
LOAD_NOISE_STD = 20         # 实际负荷随机噪声标准差

# 主程序：生成数据并训练模型
years = list(range(2024, 2034))  # 年份范围扩展至 2024-2033 年
for year in years:
    # Step 1: 生成该年的时间序列（5分钟间隔）
    start_time = f"{year}-01-01 00:00:00"
    end_time = f"{year}-12-31 23:55:00"
    # 使用 pandas.date_range 生成从当年1月1日00:00到12月31日23:55（每5分钟一个点）的时间索引
    time_index = pd.date_range(start=start_time, end=end_time, freq='5min')
    total_points = len(time_index)
    # 判断全年天数（闰年366天，平年365天）
    total_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365

    # 提取时间特征用于计算周期变化
    hours = time_index.hour + time_index.minute / 60.0          # 小时数(含分钟的小数部分)，用于日周期计算
    day_of_year = time_index.dayofyear                          # 当年中的第几天 (1~365/366)
    weekday = time_index.weekday                                # 星期几 (Monday=0, Sunday=6)

    # 生成特征数据
    # 温度 (Temperature) 时间序列
    # 年周期变化（以 cos 表示，设置相位使得南半球冬季(6-8月)温度最低，夏季(12-2月)最高）
    temp_season = TEMP_SEASON_AMPLITUDE * np.cos(2 * np.pi * (day_of_year / total_days))
    # 日周期变化（以 sin 表示，正午达到日高温，午夜达到日低温）
    temp_daily = TEMP_DAILY_AMPLITUDE * np.sin(2 * np.pi * hours / 24 - np.pi/2)
    # 随机噪声
    temp_noise = np.random.normal(loc=0, scale=TEMP_NOISE_STD, size=total_points)
    # 综合得到温度序列
    temperature = TEMP_BASE + temp_season + temp_daily + temp_noise

    # 湿度 (Humidity) 时间序列
    # 年周期变化（与温度类似，假设夏季(12-2月)湿度较高，冬季(6-8月)较低）
    humid_season = HUMID_SEASON_AMPLITUDE * np.cos(2 * np.pi * (day_of_year / total_days))
    # 日周期变化（采用 cos，使得午夜湿度最高，正午最低）
    humid_daily = HUMID_DAILY_AMPLITUDE * np.cos(2 * np.pi * hours / 24)
    # 随机噪声
    humid_noise = np.random.normal(loc=0, scale=HUMID_NOISE_STD, size=total_points)
    # 综合得到湿度序列并限制在 0~100%
    humidity = HUMID_BASE + humid_season + humid_daily + humid_noise
    humidity = np.clip(humidity, 0, 100)  # 将湿度限制在合理范围内

    # 占用率 (Occupancy) 时间序列
    # 初始化占用率数组，工作日白天 (7:00-19:00) 有占用，其余时间为 0
    occupancy = np.zeros(total_points)
    # 工作日定义：weekday 0-4 为工作日，5-6 为周末
    workday_mask = (weekday < 5)  # True 表示工作日
    # 计算占用率：对于工作日早 7 点至晚 7 点，用半正弦波模拟占用率上升和下降
    # 把 hours 转为 NumPy 数组以便进行掩码运算
    day_hours = hours.to_numpy()  # 已包含小数，可直接使用
    # 占用率公式: 在 7 <= hour < 19 时，occupancy = sin((hour-7)/12 * π)，在此之外 occupancy = 0
    # 先计算所有小时对应的值，再通过掩码筛选
    occ_values = np.sin((day_hours - 7) / 12 * np.pi)
    occ_values[(day_hours < 7) | (day_hours >= 19)] = 0  # 非工作时间设为 0
    # 将周末的所有时刻占用率强制为 0（即使白天也无人）
    occ_values[weekday >= 5] = 0
    occupancy = np.clip(occ_values, 0, 1)  # 限制在 0~1 范围（取非负部分）

    # 太阳辐照度 (Solar Irradiance) 时间序列
    # 年周期变化：计算每天正午峰值辐照度和日长
    # 使用 cos 模拟：以 12 月 21 日 (~年末第 355/356 天) 为夏至 (南半球)，辐照度最大，日长最长
    offset_day = day_of_year - (total_days - 10)  # 相对于夏至 (12/21) 的天数偏移
    # 计算当天正午最大辐照度 (Max Irradiance) 以及日长 (hours of daylight)
    max_irradiance = (SUMMER_MAX_IRR + WINTER_MIN_IRR) / 2 + \
                     (SUMMER_MAX_IRR - WINTER_MIN_IRR) / 2 * np.cos(2 * np.pi * offset_day / total_days)
    daylight_hours = 12 + DAYLIGHT_VAR_HOURS * np.cos(2 * np.pi * offset_day / total_days)
    # 初始化辐照度数组
    irradiance = np.zeros(total_points)
    # 逐个时间点计算辐照度（根据日出日落时间和半正弦曲线）
    # 计算日出和日落时刻（以小时为单位，相对于当天 0 点的小时）
    sunrise_hour = 12 - daylight_hours / 2
    sunset_hour = 12 + daylight_hours / 2
    # 对于每个时间点，如果在日出和日落之间，则计算相对日间进度；否则辐照度为 0
    day_mask = (hours >= sunrise_hour) & (hours <= sunset_hour)
    # 相对日间进度 (0 = 日出, 1 = 日落)
    rel_day_progress = np.zeros(total_points)
    rel_day_progress[day_mask] = (hours[day_mask] - sunrise_hour[day_mask]) / (sunset_hour[day_mask] - sunrise_hour[day_mask])
    # 白天时，用 sin(π * progress) 计算辐照度相对于最大值的比例
    irradiance[day_mask] = max_irradiance[day_mask] * np.sin(np.pi * rel_day_progress[day_mask])
    # 添加随机噪声（按一定百分比波动），并保证不出现负值
    irr_noise_factor = np.random.normal(loc=1.0, scale=IRR_NOISE_FRAC, size=total_points)
    irradiance = irradiance * irr_noise_factor
    irradiance = np.clip(irradiance, 0, None)  # 负值截为 0（夜晚或噪声导致的负值）

    # 生成基础负荷并计算实际负荷
    # 基础负荷 = 常数 + 年周期 + 日周期 (使用 cos 和 sin 创建周期变化)
    base_season = BASE_SEASON_AMPLITUDE * np.cos(2 * np.pi * (day_of_year / total_days))  # 南半球冬季最低, 夏季最高
    base_daily = BASE_DAILY_AMPLITUDE * np.sin(2 * np.pi * hours / 24 - np.pi/2)  # 正午略高, 午夜略低
    base_load = BASE_CONST + base_season + base_daily
    # 线性叠加各特征得到实际负荷 Load
    load = base_load + W_TEMP * temperature + W_HUMID * humidity + W_OCC * occupancy + W_IRR * irradiance
    # 加入额外随机噪声以模拟未建模的波动
    load += np.random.normal(loc=0, scale=LOAD_NOISE_STD, size=total_points)

    # 构建线性回归模型并预测负荷
    # 准备特征矩阵 X 和目标变量 y（X 包含温度、湿度、占用率、辐照度四列）
    X = np.column_stack([temperature, humidity, occupancy, irradiance])
    y = load
    model = LinearRegression()
    model.fit(X, y)
    predicted = model.predict(X)

    # 将预测结果保存为 CSV 文件
    # 创建包含 Timestamp 和 Predicted 列的 DataFrame
    df_pred = pd.DataFrame({
        'Timestamp': time_index,    # 完整时间戳（YYYY-MM-DD HH:MM:SS）
        'Predicted': predicted      # 预测负荷值
    })
    # 保存为 CSV 文件（无索引列）
    csv_filename = f'{year}.csv'
    df_pred.to_csv(csv_filename, index=False)
    print(f'[{year}] 预测结果已保存至 {csv_filename} (共 {len(df_pred)} 行)。')

    # 绘制全年预测负荷折线图并保存为 PNG 文件
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_index, predicted, color='blue')
    ax.set_title(f'Predicted Load - {year}')
    ax.set_xlabel('Time (Month)')       # 横坐标标签（月份）
    ax.set_ylabel('Predicted Load')     # 纵坐标标签（预测负荷）
    # 将横轴刻度设置为按月分隔
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b 显示月份缩写 (Jan, Feb, ...)
    # 为清晰起见，将横轴范围设为全年，并在图表右侧留出空白以显示最后一个月份标签
    ax.set_xlim(time_index.min(), time_index.max())
    plt.tight_layout()
    # 保存图像为 PNG 文件，文件名形如 YYYY_full_load.png，使用较高 DPI 提高清晰度
    img_filename = f'{year}_full_load.png'
    fig.savefig(img_filename, dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[{year}] 全年负荷曲线图已保存为 {img_filename}。')

print("模拟与预测完成！")
