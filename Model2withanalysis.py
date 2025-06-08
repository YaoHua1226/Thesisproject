import pandas as pd
import numpy as np

# Model Constants
INITIAL_CAPACITY_MWH = 2.22      # Initial battery capacity (MWh)
BATTERY_MAX_POWER_MW = 1.11      # Max charge/discharge power (MW)
ROUND_TRIP_EFF     = 0.855       # Round-trip efficiency
EFF_C = EFF_D      = np.sqrt(ROUND_TRIP_EFF)  # Split efficiency into charge/discharge
INTERVAL_MIN       = 5           # Dispatch interval in minutes
WINDOW_DAYS        = 7           # Rolling window (days) for price thresholds

# Helper: distribute hourly PV generation to 5-min intervals (returns series in MWh)
def distribute_pv(hourly_df, index_5min):
    pv_series = np.zeros(len(index_5min))
    pos_map = {pd.Timestamp(ts): idx for idx, ts in enumerate(index_5min)}
    for _, row in hourly_df.iterrows():
        ts = pd.Timestamp(row["Datetime"])
        if ts in pos_map:
            start_idx = pos_map[ts]
            total_mwh = float(row["Energy_kWh"]) / 1000.0
            slots = 60 // INTERVAL_MIN  # number of 5-min intervals in an hour
            pv_series[start_idx : start_idx + slots] = total_mwh / slots
    return pv_series

# Core simulation function for one year
def simulate_battery(df, scenario="forecast"):
    global BATTERY_CAPACITY_MWH  # 每年仿真前已在外部设置好容量
    soc = 0.0
    normal_lower = 0.2 * BATTERY_CAPACITY_MWH    # 正常运行最低 SOC (20%)
    normal_upper = 0.8 * BATTERY_CAPACITY_MWH    # 正常运行最高 SOC (80%)

    # 指标初始化
    res = {
        "charge_energy": 0.0, "discharge_energy": 0.0,
        "charge_energy_grid": 0.0, "charge_energy_pv": 0.0,
        "charge_cost": 0.0, "discharge_revenue": 0.0,
        "lost_energy_eff": 0.0, "lost_energy_spill": 0.0,
        "pv_supply_load": 0.0, "grid_import": 0.0   # <- 确保这一项保留
    }
    soc_series = []  # 用于记录每个 5 分钟间隔结束时的 SOC（%）

    prices   = df["price"].values
    demand   = df["demand_mwh"].values
    pv_seq   = df["pv_mwh"].values
    low_thr0 = df["low_thr0"].values    # 滚动 20% 分位
    low_thr  = df["low_thr"].values     # 滚动 10% 分位
    high_thr0= df["high_thr0"].values   # 滚动 80% 分位
    high_thr = df["high_thr"].values    # 滚动 90% 分位
    extreme_high_thr = df["high_thr2"].values if "high_thr2" in df.columns else None

    actual_mode = (scenario == "actual")
    grid_charge_factor = 1 if actual_mode else 1.0
    min_arbitrage_diff = 30  # 最小套利价差 (AUD/MWh)

    n = len(df)
    interval_energy = BATTERY_MAX_POWER_MW * (INTERVAL_MIN / 60.0)  # 每间隔最大能量 (MWh)

    for i in range(n):
        p  = prices[i]
        d  = demand[i]
        pv = pv_seq[i]
        l0 = low_thr0[i];  l1 = low_thr[i]
        h0 = high_thr0[i]; h1 = high_thr[i]
        h2 = extreme_high_thr[i] if (actual_mode and extreme_high_thr is not None) else None

        # 1) PV 先供给本地负荷
        use_pv = min(d, pv)
        d_left = d - use_pv            # PV 供完本地后剩余负荷
        pv_ex  = pv - use_pv           # 剩余可存或卖的 PV
        res["pv_supply_load"] += use_pv
        deliver = 0.0  # 本间隔从电池放出的能量

        # ===== 高价期 (High price period) =====
        if ((not actual_mode and p >= h0) or (actual_mode and h2 is not None and p >= h2)):
            if pv_ex > 0:
                # 【1-1】如果是极高价(>=h1) 或者未来短期内无更高价格，直接卖 PV
                if p >= h1 or (i+1 < n and prices[i+1: i+1 + (6*60//INTERVAL_MIN)].max() <= p):
                    res["discharge_revenue"] += pv_ex * p
                    pv_ex = 0.0
                else:
                    # 【1-2】否则，把 PV 剩余存进电池，期待未来更高价
                    space = (BATTERY_CAPACITY_MWH - soc) if (i+1 < n and prices[i+1: i+1 + (6*60//INTERVAL_MIN)].max() > 1.2*p) else (normal_upper - soc)
                    c = min(pv_ex, interval_energy, max(0.0, space) / EFF_C)
                    pv_ex   -= c
                    stored  = c * EFF_C      # 实际存进电池的能量
                    soc    += stored
                    res["charge_energy"]    += c
                    res["charge_energy_pv"] += c
                    res["charge_cost"]      += c * p
                    res["lost_energy_eff"]  += (c - stored)
                # 【1-3】如果电池里还有能量，且当前价 >= h1，放电卖电
                if soc > 0 and p >= h1:
                    max_d = interval_energy
                    frac  = 1.0 if h1 <= h0 else min(1.0, (p - h0) / (h1 - h0))
                    deliver = min(max_d * frac, soc * EFF_D)
                    # 在“并非极高价”但依然高于 h1 时，保留 20% SOC
                    if (p < 2*h1 and p < 300) and soc > normal_lower:
                        deliver = min(deliver, (soc - normal_lower) * EFF_D)
                    if deliver > 0:
                        taken = deliver / EFF_D
                        soc   -= taken
                        res["discharge_energy"]  += deliver
                        res["discharge_revenue"] += deliver * p
                        res["lost_energy_eff"]   += (taken - deliver)
            else:
                # 【1-4】高价阶段但无 PV 剩余，直接按是否极高价来放电
                if not ((p < 2*h1 and p < 300) and (i+1 < n and prices[i+1: i+1 + (6*60//INTERVAL_MIN)].max() > p)):
                    max_d = interval_energy
                    frac  = 1.0 if h1 <= h0 else min(1.0, (p - h0) / (h1 - h0))
                    deliver = min(max_d * frac, soc * EFF_D)
                    if (p < 2*h1 and p < 300) and soc > normal_lower:
                        deliver = min(deliver, (soc - normal_lower) * EFF_D)
                    if deliver > 0:
                        taken = deliver / EFF_D
                        soc   -= taken
                        res["discharge_energy"]  += deliver
                        res["discharge_revenue"] += deliver * p
                        res["lost_energy_eff"]   += (taken - deliver)

        # 低价期
        elif ((not actual_mode and p <= l0) or (actual_mode and p <= 0)):
            extreme_low = (p <= 0)
            cap = BATTERY_CAPACITY_MWH if (p <= 0 or p < 0.5*l1) else normal_upper
            frac = 1.0 if p < 0 or l0 <= l1 else min(1.0, (l0 - p) / (l0 - l1))
            allowed = frac * interval_energy
            did_pv_charge = False

            # 先用 PV 充电（若有 PV 副产且 SOC < cap）
            if pv_ex > 0 and soc < cap:
                c = min(pv_ex, allowed, (cap - soc) / EFF_C)
                pv_ex   -= c
                stored  = c * EFF_C
                soc    += stored
                res["charge_energy"]    += c
                res["charge_energy_pv"] += c
                res["charge_cost"]      += c * p
                res["lost_energy_eff"]  += (c - stored)
                if c > 0:
                    did_pv_charge = True

            # 再考虑从电网充电（仅在非极低价或实际场景时允许）
            if not extreme_low and soc < cap:
                if actual_mode and did_pv_charge:
                    # 实际场景：如果本间隔 PV 已经充电，则优先 PV，不再网购
                    pass
                else:
                    allowed_grid = allowed * grid_charge_factor
                    c = min(allowed_grid, (cap - soc) / EFF_C)
                    stored = c * EFF_C
                    soc   += stored
                    res["charge_energy"]      += c
                    res["charge_energy_grid"] += c
                    # Grid purchase cost = c * p + c * 4（固定 O&M）
                    res["charge_cost"]       += c * p + c * 4.0
                    res["lost_energy_eff"]   += (c - stored)

            # 实际场景下，如果价格<=0，再次尝试网购（如果 PV 未用完）
            if actual_mode and extreme_low and soc < cap:
                if not did_pv_charge:
                    allowed_grid = allowed * grid_charge_factor
                    c = min(allowed_grid, (cap - soc) / EFF_C)
                    stored = c * EFF_C
                    soc   += stored
                    res["charge_energy"]      += c
                    res["charge_energy_grid"] += c
                    res["charge_cost"]       += c * p + c * 4.0
                    res["lost_energy_eff"]   += (c - stored)

        # 中等价期
        else:
            # 【如果有 PV 副产且 SOC < normal_upper，则把 PV 存进电池
            if pv_ex > 0 and soc < normal_upper:
                cap = BATTERY_CAPACITY_MWH if (i+1 < n and (prices[i+1: i+1 + (6*60//INTERVAL_MIN)].max() >= 2*h1 or
                                                            prices[i+1: i+1 + (6*60//INTERVAL_MIN)].max() >= 300)) else normal_upper
                c = min(pv_ex, interval_energy, (cap - soc) / EFF_C)
                pv_ex   -= c
                stored  = c * EFF_C
                soc    += stored
                res["charge_energy"]    += c
                res["charge_energy_pv"] += c
                res["charge_cost"]      += c * p + c * 4.0
                res["lost_energy_eff"]  += (c - stored)

            # 【如果之后有更低价格，提前放电腾容量
            if (not actual_mode) and soc > 0 and (i+1 < n and
                (prices[i+1: i+1 + (6*60//INTERVAL_MIN)].min() <= l1 or
                 p - prices[i+1: i+1 + (6*60//INTERVAL_MIN)].min() > min_arbitrage_diff)) and \
               (i+1 < n and prices[i+1: i+1 + (6*60//INTERVAL_MIN)].min() < p):
                d_max = interval_energy
                deliver = (min(d_max, soc * EFF_D) if (i+1 < n and prices[i+1: i+1 + (6*60//INTERVAL_MIN)].min() <= 0)
                           else min(d_max, (soc - normal_lower) * EFF_D))
                if deliver > 0:
                    taken = deliver / EFF_D
                    soc  -= taken
                    res["discharge_energy"]  += deliver
                    res["discharge_revenue"] += deliver * p
                    res["lost_energy_eff"]   += (taken - deliver)

            # 如果之后有更高价格，提前网购小量充电
            elif (not actual_mode) and soc < BATTERY_CAPACITY_MWH and \
                 (i+1 < n and (prices[i+1: i+1 + (6*60//INTERVAL_MIN)].max() >= h1 or
                               (prices[i+1: i+1 + (6*60//INTERVAL_MIN)].max() - p) > min_arbitrage_diff)) and \
                 (i+1 < n and prices[i+1: i+1 + (6*60//INTERVAL_MIN)].max() > p):
                cap = BATTERY_CAPACITY_MWH if (i+1 < n and (prices[i+1: i+1 + (6*60//INTERVAL_MIN)].max() >= 2*h1 or
                                                             prices[i+1: i+1 + (6*60//INTERVAL_MIN)].max() >= 300)) else normal_upper
                c   = min(interval_energy, (cap - soc) / EFF_C)
                stored = c * EFF_C
                soc   += stored
                res["charge_energy"]      += c
                res["charge_energy_grid"] += c
                res["charge_cost"]        += c * p + c * 4.0
                res["lost_energy_eff"]    += (c - stored)

        # 任何剩余 pv_ex 都算作浪费
        if pv_ex > 0:
            res["lost_energy_spill"]  += pv_ex

        # 计算并累加本间隔的购电量
        supply_from_batt = min(d_left, deliver)
        grid_import_i = d_left - supply_from_batt
        if grid_import_i < 0:
            grid_import_i = 0.0
        res["grid_import"] += grid_import_i

        # 记录当前间隔结束时的 SOC（百分比）
        soc_series.append(soc / BATTERY_CAPACITY_MWH * 100.0)

    # 年度指标汇总
    total_c = res["charge_energy"]
    total_d = res["discharge_energy"]
    cycles  = total_d / BATTERY_CAPACITY_MWH if BATTERY_CAPACITY_MWH > 0 else 0

    avg_c_p = res["charge_cost"] / total_c if total_c > 0 else 0
    avg_d_p = res["discharge_revenue"] / total_d if total_d > 0 else 0
    cycle_fixed_cost = total_d * 4.08  # 固定 O&M 成本 (AUD/MWh)

    profit = res["discharge_revenue"] - res["charge_cost"] - cycle_fixed_cost
    loss_cost = (res["lost_energy_eff"] * avg_c_p) + (res["lost_energy_spill"] * df["price"].mean())

    total_pv = pv_seq.sum()
    pv_self_rate = ((res["pv_supply_load"] + res["charge_energy_pv"]) / total_pv * 100.0) if total_pv > 1e-9 else 0.0

    baseline_no_storage = np.sum(np.maximum(demand - pv_seq, 0))
    grid_reduction_rate = ((baseline_no_storage - res["grid_import"]) / baseline_no_storage * 100.0) \
                          if baseline_no_storage > 1e-9 else 0.0

    return {
        "year": None,  # 在主循环之外再赋值
        "profit": profit,
        "total_charge_mwh": total_c,
        "total_discharge_mwh": total_d,
        "equiv_full_cycles": cycles,
        "avg_charge_price": avg_c_p,
        "avg_discharge_price": avg_d_p,
        "efficiency_loss_mwh": res["lost_energy_eff"],
        "energy_loss_cost": loss_cost,
        "pv_self_consumption_rate": pv_self_rate,
        "grid_purchase_reduction_rate": grid_reduction_rate,
        "soc_series": soc_series  # 返回完整的 SOC 时间序列（%）
    }


# Main Simulation Loop for 2024–2033
results = []
capacity = INITIAL_CAPACITY_MWH

for year in range(2024, 2034):
    # Create 5-min index for the full year
    idx = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:55", freq=f"{INTERVAL_MIN}min")
    # Prepare a DataFrame for this year
    df_year = pd.DataFrame(index=idx)

    # Load price data
    if year == 2024:
        price_file = "2024_full_year_actual_price_data.csv"  # use actual market prices for 2024
    else:
        price_file = f"{year}_full_year_forecast.csv"        # use forecast prices for 2025+ 
    price_df = pd.read_csv(price_file)
    price_df.rename(columns=lambda x: x.strip(), inplace=True)
    # Standardize price column name 
    if "RRP" in price_df.columns and "RRP(AUD/MWh)" not in price_df.columns:
        price_df.rename(columns={"RRP": "RRP(AUD/MWh)"}, inplace=True)
    if "RRP_pred" in price_df.columns and "RRP_pred(AUD/MWh)" not in price_df.columns:
        price_df.rename(columns={"RRP_pred": "RRP_pred(AUD/MWh)"}, inplace=True)

    price_df["SETTLEMENTDATE"] = pd.to_datetime(price_df["SETTLEMENTDATE"])
    # Prefer actual price if available, otherwise use predicted price
    if "RRP(AUD/MWh)" in price_df.columns:
        price_series = price_df.set_index("SETTLEMENTDATE")["RRP(AUD/MWh)"]
    else:
        price_series = price_df.set_index("SETTLEMENTDATE")["RRP_pred(AUD/MWh)"]

    df_year["price"] = price_series.reindex(idx, method="nearest")

    # Load demand data
    load_df = pd.read_csv(f"{year}.csv")
    load_df.rename(columns=lambda x: x.strip(), inplace=True)
    if "Predicted(kW)" not in load_df.columns and "Predicted" in load_df.columns:
        load_df.rename(columns={"Predicted": "Predicted(kW)"}, inplace=True)
    load_df["Timestamp"] = pd.to_datetime(load_df["Timestamp"])
    # For 2024, we have actual vs predicted load scenarios
    demand_pred = load_df.set_index("Timestamp")["Predicted(kW)"]

    # Load PV generation data
    if year == 2024:
        pv_file_pred = "result_2024_predicted_generation.csv"
        pv_file_act  = "result_2024_mixed_generation.csv"
    else:
        pv_file_pred = f"result_{year}_predicted_generation.csv"
        pv_file_act  = None
    pv_df_pred = pd.read_csv(pv_file_pred)
    pv_df_pred["Datetime"] = pd.to_datetime(pv_df_pred["Datetime"])
    pv_df_pred["Energy_kWh"] = pd.to_numeric(pv_df_pred["Energy_kWh"], errors="coerce")
    pv_df_pred.loc[pv_df_pred["Energy_kWh"] < 0, "Energy_kWh"] = 0.0

    # Special handling for 2024
    if year == 2024:
        # Actual scenario for 2024
        pv_df_act = pd.read_csv(pv_file_act)
        pv_df_act["Datetime"] = pd.to_datetime(pv_df_act["Datetime"])
        pv_df_act["Energy_kWh"] = pd.to_numeric(pv_df_act["Energy_kWh"], errors="coerce")
        pv_df_act.loc[pv_df_act["Energy_kWh"] < 0, "Energy_kWh"] = 0.0

        df_act = pd.DataFrame(index=idx)
        df_act["price"] = df_year["price"]
        if "Actual(kW)" in load_df.columns:
            demand_act = load_df.set_index("Timestamp")["Actual(kW)"]
        else:
            demand_act = demand_pred
        df_act["demand_mwh"] = demand_act.reindex(idx, method="nearest") * (INTERVAL_MIN/60.0)/1000
        df_act["pv_mwh"]     = distribute_pv(pv_df_act, idx)
        df_act.loc[df_act["pv_mwh"] < 0, "pv_mwh"] = 0.0
        # Compute rolling price thresholds
        window = WINDOW_DAYS * (24*60 // INTERVAL_MIN)
        df_act["low_thr0"]  = df_act["price"].shift(1).rolling(window, min_periods=1).quantile(0.2)
        df_act["low_thr"]   = df_act["price"].shift(1).rolling(window, min_periods=1).quantile(0.1)
        df_act["high_thr0"] = df_act["price"].shift(1).rolling(window, min_periods=1).quantile(0.8)
        df_act["high_thr"]  = df_act["price"].shift(1).rolling(window, min_periods=1).quantile(0.9)
        # Simulate with initial capacity
        BATTERY_CAPACITY_MWH = INITIAL_CAPACITY_MWH
        result_act = simulate_battery(df_act)        # actual scenario simulation
        result_act["year"] = "2024_actual"
        results.append(result_act)

        # 2024 forecast scenario
        price_df_fc = pd.read_csv("2024_full_year_forecast.csv")
        price_df_fc.rename(columns=lambda x: x.strip(), inplace=True)
        if "RRP" in price_df_fc.columns and "RRP(AUD/MWh)" not in price_df_fc.columns:
            price_df_fc.rename(columns={"RRP": "RRP(AUD/MWh)"}, inplace=True)
        if "RRP_pred" in price_df_fc.columns and "RRP_pred(AUD/MWh)" not in price_df_fc.columns:
            price_df_fc.rename(columns={"RRP_pred": "RRP_pred(AUD/MWh)"}, inplace=True)
        price_df_fc["SETTLEMENTDATE"] = pd.to_datetime(price_df_fc["SETTLEMENTDATE"])
        if "RRP(AUD/MWh)" in price_df_fc.columns:
            price_series_fc = price_df_fc.set_index("SETTLEMENTDATE")["RRP(AUD/MWh)"]
        else:
            price_series_fc = price_df_fc.set_index("SETTLEMENTDATE")["RRP_pred(AUD/MWh)"]

        df_fc = pd.DataFrame(index=idx)
        df_fc["price"] = price_series_fc.reindex(idx, method="nearest")
        df_fc["demand_mwh"] = demand_pred.reindex(idx, method="nearest") * (INTERVAL_MIN/60.0)/1000
        df_fc["pv_mwh"]     = distribute_pv(pv_df_pred, idx)
        df_fc.loc[df_fc["pv_mwh"] < 0, "pv_mwh"] = 0.0
        df_fc["low_thr0"]  = df_fc["price"].shift(1).rolling(window, min_periods=1).quantile(0.2)
        df_fc["low_thr"]   = df_fc["price"].shift(1).rolling(window, min_periods=1).quantile(0.1)
        df_fc["high_thr0"] = df_fc["price"].shift(1).rolling(window, min_periods=1).quantile(0.8)
        df_fc["high_thr"]  = df_fc["price"].shift(1).rolling(window, min_periods=1).quantile(0.9)
        BATTERY_CAPACITY_MWH = INITIAL_CAPACITY_MWH
        result_fc = simulate_battery(df_fc)
        result_fc["year"] = "2024_forecast"
        results.append(result_fc)
        # Update battery capacity after 2024 (capacity degradation or warranty schedule)
        capacity = INITIAL_CAPACITY_MWH * 0.94
        continue

    # For years 2025–2033 (forecast scenarios)
    df_year["demand_mwh"] = demand_pred.reindex(idx, method="nearest") * (INTERVAL_MIN/60.0)/1000
    df_year["pv_mwh"] = distribute_pv(pv_df_pred, idx)
    df_year.loc[df_year["pv_mwh"] < 0, "pv_mwh"] = 0.0
    window = WINDOW_DAYS * (24*60 // INTERVAL_MIN)
    df_year["low_thr0"]  = df_year["price"].shift(1).rolling(window, min_periods=1).quantile(0.2)
    df_year["low_thr"]   = df_year["price"].shift(1).rolling(window, min_periods=1).quantile(0.1)
    df_year["high_thr0"] = df_year["price"].shift(1).rolling(window, min_periods=1).quantile(0.8)
    df_year["high_thr"]  = df_year["price"].shift(1).rolling(window, min_periods=1).quantile(0.9)

    # Set battery capacity for this year’s simulation
    BATTERY_CAPACITY_MWH = capacity
    result = simulate_battery(df_year)
    result["year"] = str(year)
    results.append(result)

    # Update capacity for next year based on warranty degradation schedule
    if year == 2025:
        capacity = INITIAL_CAPACITY_MWH * 0.90
    elif year == 2026:
        capacity = INITIAL_CAPACITY_MWH * 0.87
    elif year == 2027:
        capacity = INITIAL_CAPACITY_MWH * 0.85
    elif year == 2028:
        capacity = INITIAL_CAPACITY_MWH * 0.825
    elif 2029 <= year <= 2033:
        fraction = 0.825 - 0.005 * (year - 2028)
        capacity = INITIAL_CAPACITY_MWH * fraction
    elif 2034 <= year <= 2038:
        fraction = 0.78 - 0.02 * (year - 2034)
        capacity = INITIAL_CAPACITY_MWH * fraction
    else:
        capacity = INITIAL_CAPACITY_MWH * 0.70

# Save and display summarized results
df_results = pd.DataFrame(results)
df_results = df_results[[
    "year", "profit", "total_charge_mwh", "total_discharge_mwh",
    "equiv_full_cycles", "avg_charge_price", "avg_discharge_price",
    "efficiency_loss_mwh", "energy_loss_cost",
    "pv_self_consumption_rate", "grid_purchase_reduction_rate"
]]
df_results.columns = [
    "Year", "Profit in AUD", "Total charge energy (MWh)",
    "Total discharge energy (MWh)", "Equiv full cycles",
    "Avg charge price (AUD/MWh)", "Avg discharge price (AUD/MWh)",
    "Efficiency loss in MWh (MWh)", "Energy loss cost (AUD)",
    "PV self-consumption (%)", "Grid purchase reduction (%)"
]
print(df_results.to_string(index=False, float_format="%.2f"))
df_results.to_csv("Simulation_results.csv", index=False)

# Plot yearly profit bar chart
import matplotlib.pyplot as plt
df_new = pd.read_csv("Simulation_results.csv")
mask = ~df_new["Year"].astype(str).str.contains("actual")
profits = df_new[mask]["Profit in AUD"].values
investment = 2050000.0  # total investment in AUD
cash_flows = [-investment] + profits.tolist()
import numpy_financial as npf
irr_value = npf.irr(cash_flows)
npv_value = profits.sum() - investment
payback_year = None
cumulative = 0.0
for yr, prof in zip(df_new[mask]["Year"], profits):
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
    yr_label = str(payback_year)
    if "_" in yr_label:
        base = yr_label.split("_")[0]
        suffix = " (forecast)" if "forecast" in yr_label else " (actual)" if "actual" in yr_label else ""
        yr_label = base + suffix
    print(f"Payback period: {yr_label}")
else:
    print("Payback period: beyond 2033")

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

# Yearly Trend chart – Daily average SOC for each year
fig3, ax3 = plt.subplots(figsize=(12, 6))
for entry in results:
    year = str(entry["year"])
    if "actual" in year:
        continue  # only forecast years
    label = year.replace("_forecast", "").replace("_actual", "")
    soc_series = entry.get("soc_series")
    if soc_series is None:
        continue
    n_intervals = len(soc_series)
    points_per_day = 24 * 60 // INTERVAL_MIN  # number of 5-min intervals in a day
    days = n_intervals // points_per_day
    soc_arr = np.array(soc_series).reshape(days, points_per_day)
    daily_avg_soc = soc_arr.mean(axis=1)  # average SOC each day
    x_days = np.arange(1, days + 1)       # day-of-year index (1…365/366)
    ax3.plot(x_days, daily_avg_soc, label=label)
ax3.set_xlabel("Day of Year", fontsize=11)
ax3.set_ylabel("Average State of Charge (%)", fontsize=11)
ax3.set_title("Daily Average SOC (2024–2033)", fontsize=12)
ax3.legend()
ax3.set_ylim(0, 100)
plt.tight_layout()
plt.show()

# Daily Profile chart – Average intra-day SOC profile for each year
fig4, ax4 = plt.subplots(figsize=(12, 6))
for entry in results:
    year = str(entry["year"])
    if "actual" in year:
        continue
    label = year.replace("_forecast", "").replace("_actual", "")
    soc_series = entry.get("soc_series")
    if soc_series is None:
        continue
    n_intervals = len(soc_series)
    points_per_day = 24 * 60 // INTERVAL_MIN  # 288 intervals per day for 5-min data
    days = n_intervals // points_per_day
    soc_arr = np.array(soc_series).reshape(days, points_per_day)
    daily_profile_soc = soc_arr.mean(axis=0)  # average SOC at each time-of-day
    x_hours = np.arange(points_per_day) * (INTERVAL_MIN / 60.0)  # in hours from 0 to 24
    ax4.plot(x_hours, daily_profile_soc, label=label)
ax4.set_xlabel("Hour of Day", fontsize=11)
ax4.set_ylabel("Average State of Charge (%)", fontsize=11)
ax4.set_title("Average Daily SOC Profile (2024–2033)", fontsize=12)
ax4.legend()
ax4.set_ylim(0, 100)
plt.tight_layout()
plt.show()


