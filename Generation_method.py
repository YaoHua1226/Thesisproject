import math
import requests
import pandas as pd
from io import StringIO
from datetime import datetime

class SolarPVModel:
    def __init__(self, latitude, longitude, capacity_kw, module_eff, temp_coeff,
                 tilt_deg, azimuth_deg, inverter_eff=0.96, noct=45):
        """
        初始化光伏模型参数。
        latitude, longitude: 地理坐标（纬度, 经度）
        capacity_kw: 光伏直流装机容量（kW）
        module_eff: 组件效率（STC条件下）
        temp_coeff: 温度系数（每升高1℃功率变化率, 例 -0.0045/℃）
        tilt_deg: 组件倾角（度, 相对水平面）
        azimuth_deg: 组件方位角（度, 0°=朝正北，90°=朝正东，以顺时针为正方向）
        inverter_eff: 逆变器效率（默认0.96）
        noct: 名义工作温度NOCT（℃，默认45℃）
        """
        self.lat = math.radians(latitude)      # 纬度转换为弧度
        self.lon = longitude
        self.capacity_kw = capacity_kw
        self.module_eff = module_eff
        self.temp_coeff = temp_coeff
        self.tilt = math.radians(tilt_deg)     # 倾角弧度
        # 将方位角转换为与地理方位对应的弧度。
        # 设定0°为朝北，则正东为90°，正南180°。这里假定面板在南半球朝北(0°)。
        self.azimuth = math.radians(azimuth_deg)
        self.inverter_eff = inverter_eff
        self.noct = noct
        # 计算光伏阵列总面积（平方米）
        self.area = (capacity_kw * 1000) / (module_eff * 1000)  # = capacity_W / (module_eff * 1000)
        # 简化：上式可化简为 capacity_kw，因为capacity_kw*1000/module_eff/1000 = capacity_kw/module_eff，
        # 但直接这样写清晰表明含义。
        # 实际面积 = 容量(瓦) / (效率×1000)
        # 使用STC下1000W/m2计算面积，使得在STC条件下P_dc=capacity。
    
    def fetch_power_data(self, year):
        """
        从 NASA POWER API 获取指定年份逐小时 GHI 和气温数据，返回 pandas DataFrame。
        """
        # 设置 API 请求 URL
        params = "ALLSKY_SFC_SW_DWN,T2M"
        community = "RE"  # 可选社区：可用 Renewable Energy (RE) 数据集
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        url = (f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
               f"parameters={params}&community={community}&longitude={self.lon}&latitude={math.degrees(self.lat)}"
               f"&start={start_date}&end={end_date}&format=CSV")
        # 请求数据
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"数据获取失败，HTTP状态码: {response.status_code}")
        data_str = response.text
        
        # NASA POWER CSV 通常包含一些元数据行，需定位到数据头
        lines = data_str.splitlines()
        header_index = 0
        for i, line in enumerate(lines):
            # 找到以 "YEAR," 开头的行（假设数据列以YEAR开头），这是表头行
            if line.strip().upper().startswith("YEAR") or line.strip().startswith("\"YEAR\""):
                header_index = i
                break
        # 读入CSV（从表头行开始）
        data_csv = "\n".join(lines[header_index:])
        df = pd.read_csv(StringIO(data_csv))
        
        # 将年、月、日、小时合并为datetime（假设CSV含 YEAR, MO, DY, HR 列）
        # 有些NASA输出小时可能用0-23表示当日小时
        if {'YEAR','MO','DY','HR'}.issubset(df.columns):
            # NASA LST时间标准下，小时可能范围0-23
            # 构造时间字符串方便起见
            df['datetime'] = pd.to_datetime({
    'year':  df['YEAR'],
    'month': df['MO'],
    'day':   df['DY'],
    'hour':  df['HR']
})

            # 如果小时列是1-24，可能需要减1小时，或NASA已处理为0-23，我们这里假设为0-23
        else:
            # 若API直接提供带时刻的列，例如日期时间字符串或单列分钟序号，这里根据具体格式解析
            # 为稳妥，直接尝试解析整行日期时间
            df['datetime'] = pd.to_datetime(df.iloc[:,0], errors='coerce')
        # 丢弃无用列，只保留 datetime 和所需参数
        # 参数名称为 NASA 参数名，我们重命名为更直观列名
        df.rename(columns={
            "ALLSKY_SFC_SW_DWN": "GHI",
            "T2M": "Temp"
        }, inplace=True)
        # 只保留我们需要的列
        df = df[['datetime', 'GHI', 'Temp']]
        return df

    def _sin_solar_altitude(self, time):
        """
        计算给定时间的太阳高度角的正弦值 (sin(α))。
        time: datetime 对象（假定为当地太阳时，即12:00为真太阳正午）。
        """
        # 日序号 (1-365)
        d = time.timetuple().tm_yday
        # 计算太阳赤纬角 δ（单位：弧度）
        # δ ≈ 23.45° * sin[360/365 * (284 + d)]，转换为弧度计算
        delta = math.radians(23.45) * math.sin(math.radians((360/365) * (284 + d)))
        # 计算日时角 ω（弧度）。当地太阳时下，12:00对应 ω=0，每小时15°。
        hour = time.hour + time.minute/60.0
        omega = math.radians((hour - 12) * 15)
        # 太阳高度角 α 满足：sinα = sinφ sinδ + cosφ cosδ cosω
        sin_alpha = math.sin(self.lat) * math.sin(delta) + math.cos(self.lat) * math.cos(delta) * math.cos(omega)
        # 若太阳在地平线下，则sinα为负，将其置0（夜间无太阳）
        if sin_alpha < 0:
            sin_alpha = 0.0
        return sin_alpha
    
    def compute_poa_irradiance(self, ghi, time):
        """
        将水平面辐照度 GHI 转换为当前倾斜组件平面的 POA 辐照度。
        ghi: 当小时全球水平辐照度 (W/m^2)
        time: datetime 对象（对应此 ghi 的时刻）
        返回倾斜面辐照度 (W/m^2)。
        """
        # 太阳高度角正弦值
        sin_alpha = self._sin_solar_altitude(time)
        if sin_alpha <= 0:
            return 0.0  # 夜间或太阳在背面，辐照度为0（忽略漫射背照）
        # 计算太阳高度角余弦值
        cos_alpha = math.sqrt(1 - sin_alpha**2)
        # 计算 sin(α+β)
        sin_alpha_plus_beta = sin_alpha * math.cos(self.tilt) + cos_alpha * math.sin(self.tilt)
        # 若太阳方向偏离面板朝向导致 sin(α+β) < 0（太阳在组件背面），则取0
        if sin_alpha_plus_beta <= 0:
            return 0.0
        # 应用简化模型公式：POA = GHI * sin(α+β) / sin(α)
        poa = ghi * (sin_alpha_plus_beta / sin_alpha)
        return poa

    def compute_power(self, df):
        """
        根据输入的数据框(df)计算每个时间点的光伏发电量(kWh)，并返回包含结果的新 DataFrame。
        df 需要包含列: 'datetime', 'GHI', 'Temp'
        """
        results = []
        for _, row in df.iterrows():
            time = row['datetime']
            ghi = row['GHI']
            temp_air = row['Temp']
            # 1. 计算倾斜面辐照度
            poa = self.compute_poa_irradiance(ghi, time)
            # 2. 计算不考虑温度修正的直流功率 (W)
            p_dc_stc = poa * self.area * self.module_eff  # area(m^2)*eff*poa(kW/m2)*1000 -> W
            # 注意：poa此时是W/m^2，需要转为kW/m^2再乘面积(m^2)得到kW，再*1000为W。
            # 亦可直接：p_dc_stc = poa(W/m2) * 面积(m2) * 效率 -> W (因为效率已经按1000W/m2标定)
            # 为避免混淆，这里将poa视为W/m2，乘面积得W，再乘效率。
            # 3. 组件温度估算 (℃)
            t_cell = temp_air + (self.noct - 20) / 800.0 * poa
            # 4. 温度修正系数
            temp_factor = 1 + self.temp_coeff * (t_cell - 25)
            # 5. 修正后的直流功率 (W)
            p_dc = p_dc_stc * temp_factor
            # 6. 交流功率 (W)
            p_ac = p_dc * self.inverter_eff
            # 7. 转换为该时段发电量 (kWh)。逐小时数据每小时发电量 = 平均功率(kW) * 1h
            energy_kwh = p_ac / 1000.0  # W 转换为 kW，即相当于 kWh
            # 保存结果
            results.append({
                'datetime': time,
                'Energy_kWh': energy_kwh
            })
        result_df = pd.DataFrame(results)
        return result_df

    def run_year_simulation(self, year, csv_path=None):
        """
        运行指定年份的光伏发电量模拟：
        1. 获取NASA数据，2. 计算发电量，3. 保存CSV（如果提供路径），4. 返回结果 DataFrame。
        """
        # 获取气象数据
        df_met = self.fetch_power_data(year)
        # 计算发电量序列
        df_power = self.compute_power(df_met)
        # 保存结果
        if csv_path:
            df_power.to_csv(csv_path, index=False)
        return df_power

# 使用模型模拟2025年全年发电量
if __name__ == "__main__":
    # 初始化模型参数
    model = SolarPVModel(
        latitude=-27.5,    # 纬度
        longitude=153.0,   # 经度
        capacity_kw=2300,  # 2.3 MW = 2300 kW
        module_eff=0.147,  # 14.7% 效率
        temp_coeff=-0.0045, # 温度系数 -0.45%/℃
        tilt_deg=26,       # 倾角 26°
        azimuth_deg=0,     # 方位 0° (朝北)
        inverter_eff=0.96, # 逆变器效率
        noct=45            # 名义工作温度 45℃
    )
    # 运行模拟并保存结果
    result_df = model.run_year_simulation(2025, csv_path="StLucia_Solar_2025.csv")
    print(result_df.head())  # 打印头几行结果进行验证
