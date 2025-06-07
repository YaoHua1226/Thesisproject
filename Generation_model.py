# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from Generation_model import SolarPVModel

# 设置 matplotlib 字体及防止负号显示为方块
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 读取并合并 2021-2025年 (1-4月) 的 NASA POWER 数据
df_list = []
for year in [2021,2022,2023,2024]:
    df_year = pd.read_csv(f"nasa_power_{year}.csv", parse_dates=["Datetime"])
    df_year.rename(columns={"ALLSKY_SFC_SW_DWN":"GHI","T2M":"Temp"}, inplace=True)
    df_list.append(df_year)
# 2025 Jan–Apr 实际数据
df_2025_JanApr = pd.read_csv("nasa_power_2025_JanApr.csv", parse_dates=["Datetime"])
df_2025_JanApr.rename(columns={"ALLSKY_SFC_SW_DWN":"GHI","T2M":"Temp"}, inplace=True)
df_list.append(df_2025_JanApr)
# 合并所有数据
df_all = pd.concat(df_list, ignore_index=True)
df_all.dropna(subset=["Datetime"], inplace=True)
# 提取年月日小时
for col in ['Month','Day','Hour']:
    df_all[col] = getattr(df_all['Datetime'].dt, col.lower())

# 2. 计算典型日（按月-日-小时对 GHI、Temp 求平均）
df_typical = df_all[df_all['GHI'] != -999.0]
typical = df_typical.groupby(['Month','Day','Hour'], as_index=False)[['GHI','Temp']].mean()

# 3. 处理 2024 年数据：全年实际（仅填补 -999），及预测数据集
# 提取 2024 全年观测
df2024 = df_all[df_all['Datetime'].dt.year == 2024].copy()
# 合并典型日以获取预测值
df2024 = df2024.merge(typical, on=['Month','Day','Hour'], how='left', suffixes=('','_pred'))
# 填补缺失值
df2024['GHI'] = df2024.apply(lambda r: r['GHI_pred'] if r['GHI']==-999.0 else r['GHI'], axis=1)
df2024['Temp'] = df2024.apply(lambda r: r['Temp_pred'] if r['Temp']==-999.0 else r['Temp'], axis=1)
# 清洗后实际数据集
df2024_clean = df2024[['Datetime','GHI','Temp']].copy()

# 构建 2024 全年预测数据集
time_index_2024 = pd.date_range("2024-01-01 00:00:00","2024-12-31 23:00:00",freq="h")
df2024_pred = pd.DataFrame({'Datetime':time_index_2024})
# 提取日期字段
df2024_pred['Month'] = df2024_pred['Datetime'].dt.month
df2024_pred['Day']   = df2024_pred['Datetime'].dt.day
df2024_pred['Hour']  = df2024_pred['Datetime'].dt.hour
# 合并典型日
df2024_pred = df2024_pred.merge(typical, on=['Month','Day','Hour'], how='left')
# 重命名预测列
df2024_pred.rename(columns={'GHI':'GHI_pred','Temp':'Temp_pred'}, inplace=True)
df2024_predicted = df2024_pred[['Datetime','GHI_pred','Temp_pred']].copy()
df2024_predicted.rename(columns={'GHI_pred':'GHI','Temp_pred':'Temp'}, inplace=True)

# 4. 构建 2025 全年预测数据集
time_index_2025 = pd.date_range("2025-01-01 00:00:00","2025-12-31 23:00:00",freq="h")
df2025 = pd.DataFrame({'Datetime':time_index_2025})
for col in ['Month','Day','Hour']:
    df2025[col] = getattr(df2025['Datetime'].dt, col.lower())
# 合并典型日
df2025 = df2025.merge(typical, on=['Month','Day','Hour'], how='left')
df2025.rename(columns={'GHI':'GHI_pred','Temp':'Temp_pred'}, inplace=True)
df2025_predicted = df2025[['Datetime','GHI_pred','Temp_pred']].copy()
df2025_predicted.rename(columns={'GHI_pred':'GHI','Temp_pred':'Temp'}, inplace=True)

# 5. 发电量计算
model = SolarPVModel(
    latitude=-27.5, longitude=153.0,
    capacity_kw=2300, module_eff=0.147, temp_coeff=-0.0045,
    tilt_deg=26, azimuth_deg=0, inverter_eff=0.96, noct=45
)
res_2024_mixed = model.compute_power(df2024_clean.rename(columns={'Datetime':'datetime'})[['datetime','GHI','Temp']])
res_2024_pred  = model.compute_power(df2024_predicted.rename(columns={'Datetime':'datetime'})[['datetime','GHI','Temp']])
res_2025_pred  = model.compute_power(df2025_predicted.rename(columns={'Datetime':'datetime'})[['datetime','GHI','Temp']])
# 输出CSV
for df,fn in [(res_2024_mixed,'result_2024_mixed.csv'),(res_2024_pred,'result_2024_predicted.csv'),(res_2025_pred,'result_2025_predicted.csv')]:
    df.rename(columns={'datetime':'Datetime'}, inplace=True)
    df.to_csv(fn,index=False)

# 6. 绘制图表：保持英文字体和风格
# 6.1 2024 年 Temperature 对比 (Actual vs Predicted, Full Year)
plt.figure(figsize=(8,5))
plt.plot(df2024_clean['Datetime'], df2024_clean['Temp'], label='Actual (Filled Real Data)')
plt.plot(df2024_predicted['Datetime'], df2024_predicted['Temp'], label='Predicted (Typical Day)')
plt.title('2024 Temperature Comparison (Actual vs Predicted, Full Year)')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.savefig('temp_2024_full_comparison_en.png')
plt.show()

# 6.2 2024 年 GHI 对比 (Actual vs Predicted, Full Year)
plt.figure(figsize=(8,5))
plt.plot(df2024_clean['Datetime'], df2024_clean['GHI'], label='Actual (Filled Real Data)')
plt.plot(df2024_predicted['Datetime'], df2024_predicted['GHI'], label='Predicted (Typical Day)')
plt.title('2024 GHI Comparison (Actual vs Predicted, Full Year)')
plt.xlabel('Date')
plt.ylabel(r'GHI (W/m$^2$)')
plt.legend()
plt.savefig('ghi_2024_full_comparison_en.png')
plt.show()

# 6.3 2025 年温度对比 (实际(1–4月)+预测(5–12月) vs 全年预测) - 按小时精度
# 6.3 2025 Temperature Comparison (Jan–Apr Actual + May–Dec Predicted vs Full-Year Predicted) – hourly resolution
df2025_JanApr = pd.read_csv("nasa_power_2025_JanApr.csv", parse_dates=["Datetime"])
df2025_JanApr.rename(columns={"ALLSKY_SFC_SW_DWN": "GHI", "T2M": "Temp"}, inplace=True)

# 用预测数据填补 2025 年1–4月实际温度中的缺失值 (Temp = -999)
temp_pred_2025 = df2025_predicted[['Datetime', 'Temp']].copy()
actual_temp_2025_filled = df2025_JanApr.merge(temp_pred_2025, on='Datetime', how='left', suffixes=('_actual', '_pred'))
actual_temp_2025_filled['Temp'] = actual_temp_2025_filled['Temp_actual'].where(
    actual_temp_2025_filled['Temp_actual'] != -999.0,
    actual_temp_2025_filled['Temp_pred']
)
actual_temp_2025_filled.drop(columns=['Temp_actual', 'Temp_pred'], inplace=True)

# 拼接填补后的 1–4月实际温度和 5–12月预测温度，得到全年温度序列
actual_temp_2025_filled['month'] = actual_temp_2025_filled['Datetime'].dt.month
actual_temp_2025_filled = actual_temp_2025_filled[actual_temp_2025_filled['month'] <= 4]
df2025_predicted['month'] = df2025_predicted['Datetime'].dt.month
pred_temp_2025 = df2025_predicted.copy()
pred_temp_2025_MayDec = pred_temp_2025[pred_temp_2025['month'] >= 5]
combined_temp_2025 = pd.concat([actual_temp_2025_filled, pred_temp_2025_MayDec], ignore_index=True)
combined_temp_2025 = combined_temp_2025.sort_values('Datetime')  # 2025 年全年温度序列 (1–4月实际 + 5–12月预测)

# 绘制 2025 年温度对比曲线 (实际 Jan–Apr + 预测 May–Dec vs 全年预测)
plt.figure(figsize=(8,5))
plt.plot(combined_temp_2025['Datetime'], combined_temp_2025['Temp'], label='Actual (Jan–Apr) + Predicted (May–Dec)')
plt.plot(pred_temp_2025['Datetime'], pred_temp_2025['Temp'], label='Predicted (Full Year)')
plt.title('2025 Temperature Comparison (Jan–Apr Actual + May–Dec Predicted vs Full-Year Predicted)')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.savefig('temp_2025_full_comparison_en.png')
plt.show()


# 6.4 2025 年 GHI 预测 (Full Year)
plt.figure(figsize=(8,5))
plt.plot(df2025_predicted['Datetime'], df2025_predicted['GHI'], label='Predicted')
plt.title('2025 GHI Prediction (Full Year)')
plt.xlabel('Date')
plt.ylabel(r'GHI (W/m$^2$)')
plt.legend()
plt.savefig('ghi_2025_predicted_en.png')
plt.show()
