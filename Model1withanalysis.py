import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt


INITIAL_CAPACITY_MWH = 2.22        # 初始电池容量 (MWh)
BATTERY_MAX_POWER_MW = 1.11        # 电池最大充/放电功率 (MW)
ROUND_TRIP_EFF = 0.855             # 总往返效率
EFF_C = EFF_D = np.sqrt(ROUND_TRIP_EFF)  # 把充/放电效率各取根号
INTERVAL_MIN = 5                   # 调度间隔（分钟）
WINDOW_DAYS = 30                   # 滚动窗口天数，用于电价阈值计算


def distribute_pv(hourly_df, index_5min):
    # 新建数组，全为 0，长度等于 index_5min
    pv_series = np.zeros(len(index_5min))
    df = hourly_df.copy()
    df['Datetime_floor'] = pd.to_datetime(df['Datetime']).dt.floor('h')
    df['MWh_per_5min'] = pd.to_numeric(df['Energy_kWh']) / 1000.0 / (60 / INTERVAL_MIN)

    # 生成一个字典
    # 如果同一个小时有多条记录，则累加
    pv_map = {}
    for _, row in df.iterrows():
        ts_hour = row['Datetime_floor']
        # 如果已经存在同一小时的条目，就累加
        if ts_hour in pv_map:
            pv_map[ts_hour] += row['MWh_per_5min']
        else:
            pv_map[ts_hour] = row['MWh_per_5min']

    # 遍历 index_5min，将对应小时的发电量填入每个 5 分钟槽
    for i, ts_5min in enumerate(index_5min):
        ts_hour_floor = ts_5min.floor('h')
        if ts_hour_floor in pv_map:
            pv_series[i] = pv_map[ts_hour_floor]

    return pv_series

def simulate_battery(df):
    global BATTERY_CAPACITY_MWH
    soc = 0.0
    normal_lower = 0.1 * BATTERY_CAPACITY_MWH
    normal_upper = 0.9 * BATTERY_CAPACITY_MWH

    results = {
        "charge_energy": 0.0,       # 电网+PV 充电总量 (MWh)
        "discharge_energy": 0.0,    # 放电总量 (MWh)
        "charge_energy_grid": 0.0,  # 来自电网的充电量 (MWh)
        "charge_energy_pv": 0.0,    # 来自 PV 的充电量 (MWh)
        "discharge_revenue": 0.0,   # 放电收入 (AUD)
        "charge_cost": 0.0,         # 充电成本 (AUD)
        "lost_energy_eff": 0.0,     # 因效率损失的电量 (MWh)
        "lost_energy_spill": 0.0,   # 因 PV 弃光造成的电量 (MWh)
        "pv_supply_load": 0.0,      # PV 直接供给本地负荷的电量 (MWh)
        "grid_import": 0.0          # 本地剩余负荷由电网购电的总量 (MWh)
    }
    soc_series = []

    # 逐行遍历全年 5 分钟数据
    for _, row in df.iterrows():
        price = row['price']
        low_thr0 = row['low_thr0'] if not np.isnan(row['low_thr0']) else np.inf
        low_thr  = row['low_thr']  if not np.isnan(row['low_thr'])  else np.inf
        high_thr0 = row['high_thr0'] if not np.isnan(row['high_thr0']) else -np.inf
        high_thr  = row['high_thr']  if not np.isnan(row['high_thr'])  else -np.inf

        demand = row['demand_mwh']
        pv = row['pv_mwh']

        # PV 先满足当地负荷
        pv_to_demand = min(demand, pv)
        demand_after = demand - pv_to_demand
        pv_excess = pv - pv_to_demand
        results['pv_supply_load'] += pv_to_demand  # 记录 PV 供给负荷

        # 高电价段
        if price >= high_thr0:
            # 若有 PV 余量，则优先 PV 充电，不放电
            if pv_excess > 0:
                if soc < normal_upper:
                    max_c = BATTERY_MAX_POWER_MW * (INTERVAL_MIN / 60.0)
                    space = normal_upper - soc
                    c_pv = min(pv_excess, max_c, space / EFF_C)
                    if c_pv > 1e-9:
                        pv_excess -= c_pv
                        stored = c_pv * EFF_C
                        soc += stored
                        results['charge_energy']    += c_pv
                        results['charge_energy_pv'] += c_pv
                        results['charge_cost']      += c_pv * price
                        lost = c_pv - stored
                        results['lost_energy_eff']  += lost
                        results['charge_cost']      += lost * price
            else:
                # 无 PV 余量 -> 考虑放电
                extreme_high = (price >= high_thr * 2) or (price >= 300)
                max_d = BATTERY_MAX_POWER_MW * (INTERVAL_MIN / 60.0)
                if high_thr > high_thr0:
                    fraction = 1.0 if price >= high_thr else (price - high_thr0) / (high_thr - high_thr0)
                else:
                    fraction = 1.0
                max_d *= fraction
                deliverable = 0.0
                if extreme_high:
                    deliverable = min(max_d, soc * EFF_D)
                else:
                    if soc > normal_lower:
                        deliverable = min(max_d, (soc - normal_lower) * EFF_D)
                if deliverable > 1e-9:
                    taken = deliverable / EFF_D
                    soc -= taken
                    results['discharge_energy'] += deliverable
                    # 电池放电优先满足本地负荷
                    use_for_load = min(deliverable, demand_after)
                    demand_after -= use_for_load
                    results['discharge_revenue'] += deliverable * price
                    lost = taken - deliverable
                    results['lost_energy_eff']   += lost
                    results['discharge_revenue'] -= lost * price

        # 低电价段
        elif price <= low_thr0:
            extreme_low = (price <= 0) or (price < low_thr * 0.5 if low_thr != np.inf else False)
            cap_limit = BATTERY_CAPACITY_MWH if extreme_low else normal_upper
            if soc < cap_limit:
                max_c = BATTERY_MAX_POWER_MW * (INTERVAL_MIN / 60.0)
                if price < 0:
                    fraction = 1.0
                elif low_thr0 > low_thr:
                    fraction = 1.0 if price <= low_thr else (low_thr0 - price) / (low_thr0 - low_thr)
                else:
                    fraction = 1.0
                allowed = fraction * max_c
                c_pv = 0.0
                # 先用 PV 占位充电
                if pv_excess > 0:
                    space = cap_limit - soc
                    c_pv = min(pv_excess, allowed, space / EFF_C)
                    if c_pv > 1e-9:
                        pv_excess -= c_pv
                        stored = c_pv * EFF_C
                        soc += stored
                        results['charge_energy']    += c_pv
                        results['charge_energy_pv'] += c_pv
                        results['charge_cost']      += c_pv * price
                        lost = c_pv - stored
                        results['lost_energy_eff']  += lost
                        results['charge_cost']      += lost * price
                # 若仍有空间，则从电网充电
                if soc < cap_limit:
                    rem_power = allowed - c_pv
                    space = cap_limit - soc
                    c_grid = min(rem_power, space / EFF_C)
                    if c_grid > 1e-9:
                        stored = c_grid * EFF_C
                        soc += stored
                        results['charge_energy']      += c_grid
                        results['charge_energy_grid'] += c_grid
                        results['charge_cost']        += c_grid * price
                        results['charge_cost']        += c_grid * 4.0
                        lost = c_grid - stored
                        results['lost_energy_eff']    += lost
                        results['charge_cost']        += lost * price

        # 中等电价段
        else:
            if pv_excess > 0 and soc < normal_upper:
                max_c = BATTERY_MAX_POWER_MW * (INTERVAL_MIN / 60.0)
                space = normal_upper - soc
                c_pv = min(pv_excess, max_c, space / EFF_C)
                if c_pv > 1e-9:
                    pv_excess -= c_pv
                    stored = c_pv * EFF_C
                    soc += stored
                    results['charge_energy']    += c_pv
                    results['charge_energy_pv'] += c_pv
                    results['charge_cost']      += c_pv * price
                    lost = c_pv - stored
                    results['lost_energy_eff']  += lost
                    results['charge_cost']      += lost * price

        # 剩余 PV 全部弃光
        if pv_excess > 1e-9:
            results['lost_energy_spill'] += pv_excess
            results['discharge_revenue'] -= pv_excess * price

        # 本地剩余负荷由电网购电
        grid_import_i = demand_after
        if grid_import_i < 0:
            grid_import_i = 0.0
        results['grid_import'] += grid_import_i

        # 记录 SOC (%) 到 soc_series
        soc_series.append(soc / BATTERY_CAPACITY_MWH * 100.0)

    # 年度汇总
    total_c = results['charge_energy']
    total_d = results['discharge_energy']
    cycles = total_d / BATTERY_CAPACITY_MWH if BATTERY_CAPACITY_MWH > 0 else 0
    avg_c_price = results['charge_cost'] / total_c if total_c > 1e-9 else 0
    avg_d_price = results['discharge_revenue'] / total_d if total_d > 1e-9 else 0
    cycle_fixed_cost = total_d * 4.08
    profit = results['discharge_revenue'] - results['charge_cost'] - cycle_fixed_cost
    loss_cost = results['lost_energy_eff'] * avg_c_price + results['lost_energy_spill'] * df['price'].mean()

    # PV 自耗率（%）= (pv_supply_load + charge_energy_pv) / total_pv * 100
    total_pv = df['pv_mwh'].sum()
    if total_pv > 1e-9:
        pv_self_consumption_rate = (results['pv_supply_load'] + results['charge_energy_pv']) / total_pv * 100.0
    else:
        pv_self_consumption_rate = 0.0

    # 电网购电减少率（%）
    total_demand = df['demand_mwh'].sum()
    # 若无储能，则 PV 仍先供负荷，剩余负荷全部靠电网
    baseline_no_storage = total_demand - results['pv_supply_load']
    if baseline_no_storage > 1e-9:
        grid_purchase_reduction_rate = (baseline_no_storage - results['grid_import']) / baseline_no_storage * 100.0
    else:
        grid_purchase_reduction_rate = 0.0

    return {
        "year": None,
        "profit": profit,
        "total_charge_mwh": total_c,
        "total_discharge_mwh": total_d,
        "equiv_full_cycles": cycles,
        "avg_charge_price": avg_c_price,
        "avg_discharge_price": avg_d_price,
        "efficiency_loss_mwh": results['lost_energy_eff'],
        "energy_loss_cost": loss_cost,
        "pv_self_consumption_rate": pv_self_consumption_rate,
        "grid_purchase_reduction_rate": grid_purchase_reduction_rate,
        "soc_series": soc_series
    }


# 主循环
years = range(2024, 2034)
results_list = []
capacity = INITIAL_CAPACITY_MWH

for year in years:
    # 创建全年 5 分钟索引
    index_year = pd.date_range(start=f"{year}-01-01 00:00", end=f"{year}-12-31 23:55", freq=f"{INTERVAL_MIN}min")
    scen_df = pd.DataFrame(index=index_year)

    # 导入电价预测数据
    if year == 2024:
        price_file = "2024_full_year_forecast_price_data.csv"
    else:
        price_file = f"{year}_full_year_forecast.csv"
    price_df = pd.read_csv(price_file)
    # 统一列名
    if 'RRP_pred' in price_df.columns and 'RRP_pred(AUD/MWh)' not in price_df.columns:
        price_df.rename(columns={'RRP_pred': 'RRP_pred(AUD/MWh)'}, inplace=True)
    price_df['SETTLEMENTDATE'] = pd.to_datetime(price_df['SETTLEMENTDATE'])
    scen_df['price'] = price_df.set_index('SETTLEMENTDATE')['RRP_pred(AUD/MWh)'].reindex(index_year, method='nearest')

    # 导入负荷预测数据
    demand_df = pd.read_csv(f"{year}.csv")
    demand_df.rename(columns=lambda x: x.strip(), inplace=True)
    if 'Predicted' in demand_df.columns and 'Predicted(kW)' not in demand_df.columns:
        demand_df.rename(columns={'Predicted': 'Predicted(kW)'}, inplace=True)
    demand_df['Timestamp'] = pd.to_datetime(demand_df['Timestamp'])
    scen_df['demand_mwh'] = demand_df.set_index('Timestamp')['Predicted(kW)'].reindex(index_year, method='nearest') * (INTERVAL_MIN / 60.0) / 1000.0

    # 导入 PV 生成预测数据并分配到 5 分钟
    pv_df = pd.read_csv(f"result_{year}_predicted_generation.csv")
    pv_df['Datetime'] = pd.to_datetime(pv_df['Datetime'])
    pv_df['Energy_kWh'] = pd.to_numeric(pv_df['Energy_kWh'], errors='coerce')
    pv_df.loc[pv_df['Energy_kWh'] < 0, 'Energy_kWh'] = 0.0
    scen_df['pv_mwh'] = distribute_pv(pv_df, index_year)

    # 计算滚动电价阈值
    window_size = WINDOW_DAYS * (24 * 60 // INTERVAL_MIN)
    scen_df['low_thr0']  = scen_df['price'].shift(1).rolling(window_size, min_periods=1).quantile(0.2)
    scen_df['low_thr']   = scen_df['price'].shift(1).rolling(window_size, min_periods=1).quantile(0.1)
    scen_df['high_thr0'] = scen_df['price'].shift(1).rolling(window_size, min_periods=1).quantile(0.8)
    scen_df['high_thr']  = scen_df['price'].shift(1).rolling(window_size, min_periods=1).quantile(0.9)

    # 设置当年电池容量并调用仿真
    BATTERY_CAPACITY_MWH = capacity
    result = simulate_battery(scen_df)
    result['year'] = year
    results_list.append(result)

    # 更新下一年电池容量（2024 年后逐年 97.5% 降级，最低 75%）
    if year == 2024:
        capacity *= 0.93
    else:
        capacity *= 0.975
    if capacity < INITIAL_CAPACITY_MWH * 0.75:
        capacity = INITIAL_CAPACITY_MWH * 0.75


# 汇总并输出结果表格
df_results = pd.DataFrame(results_list)
df_results = df_results[[
    "year", "profit", "total_charge_mwh", "total_discharge_mwh",
    "equiv_full_cycles", "avg_charge_price", "avg_discharge_price",
    "efficiency_loss_mwh", "energy_loss_cost",
    "pv_self_consumption_rate", "grid_purchase_reduction_rate"
]].round(2)
df_results.columns = [
    "Year", "Profit in AUD", "Total charge energy (MWh)",
    "Total discharge energy (MWh)", "Equivalent full cycles",
    "Avg charge price (AUD/MWh)", "Avg discharge price (AUD/MWh)",
    "Efficiency loss (MWh)", "Lost energy cost (AUD)",
    "PV self-consumption (%)", "Grid purchase reduction (%)"
]

print("Annual arbitrage simulation results with extended analysis:")
print(df_results.to_string(index=False, float_format="%.2f"))
df_results.to_csv("Simulation_results.csv", index=False)
print("Generated Simulation_results.csv")

# 财务分析：NPV、IRR、回收期（2024–2033）
df_new = pd.read_csv("Simulation_results.csv")
profits = df_new["Profit in AUD"].values
investment = 2050000.0  # 总投资 (AUD)
cash_flows = [-investment] + profits.tolist()

irr_value = npf.irr(cash_flows)
npv_value = profits.sum() - investment

# 计算回收期：累计利润 ≥ 投资时的那一年
payback_year = None
cumulative = 0.0
for yr, prof in zip(df_new["Year"], profits):
    cumulative += prof
    if payback_year is None and cumulative >= investment:
        payback_year = yr
        break

print(f"Undiscounted NPV (2024–2033): {npv_value:,.0f} AUD")
if irr_value is not None:
    print(f"IRR (2024–2033): {irr_value*100:.2f}%")
else:
    print("IRR (2024–2033): N/A")
if payback_year is not None:
    print(f"Payback period: {payback_year}")
else:
    print("Payback period: beyond 2033")

# 图表 1：年度利润柱状图
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(df_new["Year"], df_new["Profit in AUD"])
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 2000, f"{h:,.0f}",
            ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Profit (AUD)", fontsize=11)
ax.set_title("2024–2033 YEARS PROFIT", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout(pad=2)
plt.show()


# 图表 2：每日平均 SOC (每年一条曲线)
fig2, ax2 = plt.subplots(figsize=(12, 6))
for entry in results_list:
    year = entry["year"]
    soc_series = entry.get("soc_series")
    if soc_series is None or len(soc_series) == 0:
        continue
    n_intervals = len(soc_series)
    points_per_day = 24 * 60 // INTERVAL_MIN  # 288 段
    days = n_intervals // points_per_day
    soc_arr = np.array(soc_series).reshape(days, points_per_day)
    daily_avg_soc = soc_arr.mean(axis=1)
    ax2.plot(np.arange(1, days + 1), daily_avg_soc, label=str(year))
ax2.set_xlabel("Day of Year", fontsize=11)
ax2.set_ylabel("Average State of Charge (%)", fontsize=11)
ax2.set_title("Daily Average SOC (2024–2033)", fontsize=12)
ax2.legend()
ax2.set_ylim(0, 100)
plt.tight_layout()
plt.show()


# 图表 3：平均日内 SOC 曲线 (每年一条曲线)
fig3, ax3 = plt.subplots(figsize=(12, 6))
for entry in results_list:
    year = entry["year"]
    soc_series = entry.get("soc_series")
    if soc_series is None or len(soc_series) == 0:
        continue
    n_intervals = len(soc_series)
    points_per_day = 24 * 60 // INTERVAL_MIN  # 288 段
    days = n_intervals // points_per_day
    soc_arr = np.array(soc_series).reshape(days, points_per_day)
    daily_profile_soc = soc_arr.mean(axis=0)
    hours = np.arange(points_per_day) * (INTERVAL_MIN / 60.0)
    ax3.plot(hours, daily_profile_soc, label=str(year))
ax3.set_xlabel("Hour of Day", fontsize=11)
ax3.set_ylabel("Average State of Charge (%)", fontsize=11)
ax3.set_title("Average Daily SOC Profile (2024–2033)", fontsize=12)
ax3.legend()
ax3.set_ylim(0, 100)
plt.tight_layout()
plt.show()

