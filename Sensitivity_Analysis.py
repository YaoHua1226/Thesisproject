import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Baseline model parameters (from Model2.py)
INITIAL_CAPACITY_MWH = 2.22   # Initial battery capacity (MWh)
BATTERY_CAPACITY_MWH = INITIAL_CAPACITY_MWH  # Current battery capacity (modifiable per scenario)
BATTERY_MAX_POWER_MW = 1.11   # Max charge/discharge power (MW)
ROUND_TRIP_EFF = 0.855        # Round-trip efficiency
EFF_C = EFF_D = np.sqrt(ROUND_TRIP_EFF)  # Charge/discharge efficiency for each direction
INTERVAL_MIN = 5              # Dispatch interval in minutes
WINDOW_DAYS = 7               # Rolling window (days) for price thresholds
# Additional parameters for flexibility in scenarios
SOC_LOWER_FRAC = 0.2          # Lower SOC limit as fraction of capacity (20%)
SOC_UPPER_FRAC = 0.8          # Upper SOC limit as fraction of capacity (80%)
EXTRA_GRID_COST = 4.0         # Extra cost per MWh for grid charging (AUD)
MIN_ARBITRAGE_DIFF = 30       # Minimum arbitrage price difference (AUD/MWh)

# Helper: distribute hourly PV generation to 5-min intervals (returns a numpy array in MWh)
def distribute_pv(hourly_df, index_5min):
    pv_series = np.zeros(len(index_5min))
    # Create a lookup from hourly timestamps to index positions in the 5-min index
    pos_map = {pd.Timestamp(ts): idx for idx, ts in enumerate(index_5min)}
    for _, row in hourly_df.iterrows():
        ts = pd.Timestamp(row["Datetime"])
        if ts in pos_map:
            start_idx = pos_map[ts]
            total_mwh = float(row["Energy_kWh"]) / 1000.0  # convert kWh to MWh
            slots = 60 // INTERVAL_MIN  # number of 5-min intervals in an hour
            pv_series[start_idx : start_idx + slots] = total_mwh / slots
    return pv_series

# Core simulation function for one year (returns profit in AUD for that year)
def simulate_battery(df, scenario="forecast"):
    global BATTERY_CAPACITY_MWH
    # Initialize state-of-charge and results accumulators
    soc = 0.0  # state of charge (MWh)
    normal_lower = SOC_LOWER_FRAC * BATTERY_CAPACITY_MWH   # normal operation lower SOC (MWh)
    normal_upper = SOC_UPPER_FRAC * BATTERY_CAPACITY_MWH   # normal operation upper SOC (MWh)
    res = {
        "charge_energy": 0.0, "discharge_energy": 0.0,
        "charge_energy_grid": 0.0, "charge_energy_pv": 0.0,
        "charge_cost": 0.0, "discharge_revenue": 0.0,
        "lost_energy_eff": 0.0, "lost_energy_spill": 0.0
    }
    # Extract arrays for faster access
    prices = df["price"].values
    demand = df["demand_mwh"].values
    pv_seq = df["pv_mwh"].values
    low_thr0 = df["low_thr0"].values    # 20th percentile rolling price
    low_thr  = df["low_thr"].values     # 10th percentile rolling price
    high_thr0 = df["high_thr0"].values  # 80th percentile rolling price
    high_thr  = df["high_thr"].values   # 90th percentile rolling price
    # Extreme high price threshold (e.g., 95th percentile) if present (for actual scenario)
    extreme_high_thr = df["high_thr2"].values if "high_thr2" in df.columns else None

    actual_mode = (scenario == "actual")
    # Grid charging power factor (actual scenario might restrict grid charge rate to 50%)
    grid_charge_factor = 0.5 if actual_mode else 1.0
    # Minimum arbitrage price difference for trading (use global setting)
    min_arbitrage_diff = MIN_ARBITRAGE_DIFF

    n = len(df)
    interval_energy = BATTERY_MAX_POWER_MW * (INTERVAL_MIN / 60.0)  # max energy (MWh) per dispatch interval

    # Iterate over each 5-min interval in the year
    for i in range(n):
        p  = prices[i]
        d  = demand[i]
        pv = pv_seq[i]
        l0 = low_thr0[i];  l1 = low_thr[i]
        h0 = high_thr0[i]; h1 = high_thr[i]
        h2 = extreme_high_thr[i] if (actual_mode and extreme_high_thr is not None) else None

        # Use PV to supply demand first
        use_pv = min(d, pv)
        d_left = d - use_pv             # remaining demand after using PV
        pv_excess = pv - use_pv         # excess PV generation after meeting demand

        # Look ahead 6 hours into the future to gauge price trend
        horizon = 6 * 60 // INTERVAL_MIN  # number of intervals in 6 hours
        future_prices = prices[i+1 : min(n, i+1+horizon)]
        future_max = future_prices.max() if len(future_prices) > 0 else p
        future_min = future_prices.min() if len(future_prices) > 0 else p

        # ** High Price Period **
        if ((not actual_mode and p >= h0) or (actual_mode and h2 is not None and p >= h2)):
            if pv_excess > 0:
                # Case 1: High price with PV excess available
                if p >= h1 or future_max <= p:
                    # If price is extremely high (>=90th percentile) or no higher price expected, sell all PV excess
                    res["discharge_revenue"] += pv_excess * p
                    pv_excess = 0.0
                else:
                    # If an even higher price is expected later, charge battery with PV excess instead of selling now
                    # Determine available charging space (use full capacity if a much higher price is expected soon)
                    if future_max > 1.2 * p:
                        space = BATTERY_CAPACITY_MWH - soc
                    else:
                        space = normal_upper - soc
                    c = min(pv_excess, interval_energy, max(0.0, space) / EFF_C)

                    pv_excess -= c
                    stored = c * EFF_C
                    soc += stored
                    res["charge_energy"]    += c
                    res["charge_energy_pv"] += c
                    # Opportunity cost: consider not selling PV at price p as a "cost"
                    res["charge_cost"]     += c * p
                    res["lost_energy_eff"] += (c - stored)  # account for efficiency loss
                # After charging with PV (if any) during high price, consider discharging from battery
                if soc > 0 and p >= h1:
                    max_d = interval_energy
                    # Determine fraction of max power to discharge (scale if price is between 80th and 90th percentile)
                    frac = 1.0 if h1 <= h0 else min(1.0, (p - h0) / (h1 - h0))
                    deliver = min(max_d * frac, soc * EFF_D)
                    # If price is high but not extreme, keep battery above normal_lower (avoid full discharge)
                    if (p < 2 * h1 and p < 300) and soc > normal_lower:
                        deliver = min(deliver, (soc - normal_lower) * EFF_D)
                    if deliver > 1e-9:
                        taken = deliver / EFF_D
                        soc  -= taken
                        res["discharge_energy"]  += deliver
                        res["discharge_revenue"] += deliver * p
                        res["lost_energy_eff"]   += (taken - deliver)
            else:
                # Case 2: High price with no PV excess
                if not ((p < 2 * h1 and p < 300) and future_max > p):
                    max_d = interval_energy
                    frac = 1.0 if h1 <= h0 else min(1.0, (p - h0) / (h1 - h0))
                    deliver = min(max_d * frac, soc * EFF_D)
                    if (p < 2 * h1 and p < 300) and soc > normal_lower:
                        deliver = min(deliver, (soc - normal_lower) * EFF_D)
                    if deliver > 1e-9:
                        taken = deliver / EFF_D
                        soc  -= taken
                        res["discharge_energy"]  += deliver
                        res["discharge_revenue"] += deliver * p
                        res["lost_energy_eff"]   += (taken - deliver)

        # ** Low Price Period **
        elif ((not actual_mode and p <= l0) or (actual_mode and p <= 0)):
            # Only charge when price is sufficiently low
            extreme_low = (p <= 0)
            # If price is extremely low or far below 10th percentile, allow charging up to full capacity
            if p <= 0 or p < 0.5 * l1:
                cap_limit = BATTERY_CAPACITY_MWH
            else:
                cap_limit = normal_upper
            frac = 1.0 if (p < 0 or l0 <= l1) else min(1.0, (l0 - p) / (l0 - l1))
            allowed_energy = frac * interval_energy
            # 1) Charge from PV excess first (prioritize using any PV generation)
            did_pv_charge = False
            if pv_excess > 0 and soc < cap_limit:
                c = min(pv_excess, allowed_energy, (cap_limit - soc) / EFF_C)
                pv_excess -= c
                stored = c * EFF_C
                soc += stored
                res["charge_energy"]    += c
                res["charge_energy_pv"] += c
                res["charge_cost"]      += c * p  # cost for charging from PV (could be opportunity cost if p>0)
                res["lost_energy_eff"]  += (c - stored)
                if c > 1e-9:
                    did_pv_charge = True
            # 2) Charge from grid if allowed (forecast mode or actual mode with no PV used in this interval)
            if not extreme_low and soc < cap_limit:
                if actual_mode and did_pv_charge:
                    # In actual mode, skip grid charge if PV was used (to prioritize PV utilization)
                    pass
                else:
                    allowed_grid = allowed_energy * grid_charge_factor
                    c = min(allowed_grid, (cap_limit - soc) / EFF_C)
                    stored = c * EFF_C
                    soc += stored
                    res["charge_energy"]      += c
                    res["charge_energy_grid"] += c
                    res["charge_cost"]       += c * p + c * EXTRA_GRID_COST
                    res["lost_energy_eff"]   += (c - stored)
            # 3) In actual mode, if price is extremely low (<= 0) and PV wasn't used, charge from grid regardless
            if actual_mode and extreme_low and soc < cap_limit:
                if not did_pv_charge:
                    allowed_grid = allowed_energy * grid_charge_factor
                    c = min(allowed_grid, (cap_limit - soc) / EFF_C)
                    stored = c * EFF_C
                    soc += stored
                    res["charge_energy"]      += c
                    res["charge_energy_grid"] += c
                    res["charge_cost"]       += c * p + c * EXTRA_GRID_COST
                    res["lost_energy_eff"]   += (c - stored)

        # ** Medium Price Period **
        else:
            # 1) If PV excess exists and battery is not at normal upper limit, store PV in battery for future use
            if pv_excess > 0 and soc < normal_upper:
                cap_limit = BATTERY_CAPACITY_MWH if (future_max >= 2 * h1 or future_max >= 300) else normal_upper
                c = min(pv_excess, interval_energy, (cap_limit - soc) / EFF_C)
                pv_excess -= c
                stored = c * EFF_C
                soc += stored
                res["charge_energy"]    += c
                res["charge_energy_pv"] += c
                # Apply an O&M cost for charging this energy (grid-like cost, since battery cycling incurs cost)
                res["charge_cost"]      += c * p + c * EXTRA_GRID_COST
                res["lost_energy_eff"]  += (c - stored)
            # 2) If a much lower price is coming, preemptively discharge some energy now (forecast mode only)
            if (not actual_mode) and soc > 0 and (future_min <= l1 or (p - future_min) > min_arbitrage_diff) and future_min < p:
                d_max = interval_energy
                if future_min <= 0:
                    deliver = min(d_max, soc * EFF_D)
                else:
                    # Discharge only down to normal_lower to preserve some SOC
                    deliver = min(d_max, (soc - normal_lower) * EFF_D)
                if deliver > 1e-9:
                    taken = deliver / EFF_D
                    soc  -= taken
                    res["discharge_energy"]  += deliver
                    res["discharge_revenue"] += deliver * p
                    res["lost_energy_eff"]   += (taken - deliver)
            # 3) If a much higher price is expected, preemptively charge a bit from grid now (forecast mode only)
            elif (not actual_mode) and soc < BATTERY_CAPACITY_MWH and (future_max >= h1 or (future_max - p) > min_arbitrage_diff) and future_max > p:
                cap_limit = BATTERY_CAPACITY_MWH if (future_max >= 2 * h1 or future_max >= 300) else normal_upper
                c = min(interval_energy, (cap_limit - soc) / EFF_C)
                stored = c * EFF_C
                soc += stored
                res["charge_energy"]      += c
                res["charge_energy_grid"] += c
                res["charge_cost"]        += c * p + c * EXTRA_GRID_COST
                res["lost_energy_eff"]    += (c - stored)
        # Account for any unused PV at this interval (spilled energy)
        if pv_excess > 0:
            res["lost_energy_spill"] += pv_excess

    # Calculate yearly profit and other metrics (if needed)
    total_c = res["charge_energy"]
    total_d = res["discharge_energy"]
    cycles  = total_d / BATTERY_CAPACITY_MWH if BATTERY_CAPACITY_MWH > 0 else 0
    avg_charge_price = res["charge_cost"] / total_c if total_c > 0 else 0.0
    avg_discharge_price = res["discharge_revenue"] / total_d if total_d > 0 else 0.0
    # Include fixed O&M cost for discharged energy (e.g., 4.08 AUD per MWh discharged)
    fixed_cost = total_d * 4.08
    profit = res["discharge_revenue"] - res["charge_cost"] - fixed_cost

    return profit

# --- Data Loading for 2024 ---
# (Ensure all required CSV files are available in the working directory)
# Load 2024 price data (forecast prices used for simulation)
price_df = pd.read_csv("2024_full_year_forecast.csv")
price_df.rename(columns=lambda x: x.strip(), inplace=True)
price_df["SETTLEMENTDATE"] = pd.to_datetime(price_df["SETTLEMENTDATE"])
# Determine the price column (actual or predicted price)
if "RRP(AUD/MWh)" in price_df.columns:
    price_series = price_df.set_index("SETTLEMENTDATE")["RRP(AUD/MWh)"]
elif "RRP_pred(AUD/MWh)" in price_df.columns:
    price_series = price_df.set_index("SETTLEMENTDATE")["RRP_pred(AUD/MWh)"]
else:
    # Fallback if column names differ (e.g., "RRP")
    price_col = [col for col in price_df.columns if col.upper().startswith("RRP")][0]
    price_series = price_df.set_index("SETTLEMENTDATE")[price_col]

# Create a DateTime index at 5-minute intervals for the full year 2024
idx_2024 = pd.date_range("2024-01-01 00:00", "2024-12-31 23:55", freq=f"{INTERVAL_MIN}min")
# Initialize DataFrame for 2024 with the 5-min index
df_2024 = pd.DataFrame(index=idx_2024)
# Align price data to the 5-min index (forward-fill or nearest neighbor)
df_2024["price"] = price_series.reindex(idx_2024, method="nearest")

# Load 2024 demand data (predicted demand in kW)
load_df = pd.read_csv("2024.csv")
load_df.rename(columns=lambda x: x.strip(), inplace=True)
load_df["Timestamp"] = pd.to_datetime(load_df["Timestamp"])
# Use predicted load (kW) and convert to MWh per 5-min interval
if "Predicted(kW)" in load_df.columns:
    demand_series = load_df.set_index("Timestamp")["Predicted(kW)"]
elif "Predicted" in load_df.columns:
    demand_series = load_df.set_index("Timestamp")["Predicted"]
else:
    raise RuntimeError("Predicted load data not found in 2024.csv")
df_2024["demand_mwh"] = demand_series.reindex(idx_2024, method="nearest") * (INTERVAL_MIN/60.0) / 1000.0

# Load 2024 PV generation data (predicted PV in kWh per hour)
pv_pred_df = pd.read_csv("result_2024_predicted_generation.csv")
pv_pred_df["Datetime"] = pd.to_datetime(pv_pred_df["Datetime"])
pv_pred_df["Energy_kWh"] = pd.to_numeric(pv_pred_df["Energy_kWh"], errors="coerce").fillna(0.0)
pv_pred_df.loc[pv_pred_df["Energy_kWh"] < 0, "Energy_kWh"] = 0.0
# Distribute hourly PV generation to 5-min intervals
df_2024["pv_mwh"] = distribute_pv(pv_pred_df, idx_2024)
df_2024.loc[df_2024["pv_mwh"] < 0, "pv_mwh"] = 0.0

# Compute rolling price thresholds for baseline scenario (7-day window)
window_size = WINDOW_DAYS * (24*60 // INTERVAL_MIN)
df_2024["low_thr0"]  = df_2024["price"].shift(1).rolling(window_size, min_periods=1).quantile(0.2)
df_2024["low_thr"]   = df_2024["price"].shift(1).rolling(window_size, min_periods=1).quantile(0.1)
df_2024["high_thr0"] = df_2024["price"].shift(1).rolling(window_size, min_periods=1).quantile(0.8)
df_2024["high_thr"]  = df_2024["price"].shift(1).rolling(window_size, min_periods=1).quantile(0.9)
# (Note: high_thr2 (95th percentile) can be added if needed for actual scenario)

# Sensitivity Analysis Scenarios (2024)
results = []  # to store profit results for each scenario

# Baseline scenario: 2024 profit with all baseline parameters
BATTERY_CAPACITY_MWH = INITIAL_CAPACITY_MWH
BATTERY_MAX_POWER_MW = 1.11
ROUND_TRIP_EFF = 0.855; EFF_C = EFF_D = np.sqrt(ROUND_TRIP_EFF)
SOC_LOWER_FRAC = 0.2; SOC_UPPER_FRAC = 0.8
EXTRA_GRID_COST = 4.0
MIN_ARBITRAGE_DIFF = 30
profit_base = simulate_battery(df_2024, scenario="forecast")
results.append({"Scenario": "Baseline", "Profit (AUD)": profit_base})

# Initial Capacity = 4.44 MWh (increase from 2.22)
BATTERY_CAPACITY_MWH = 4.44
BATTERY_MAX_POWER_MW = 1.11
ROUND_TRIP_EFF = 0.855; EFF_C = EFF_D = np.sqrt(ROUND_TRIP_EFF)
SOC_LOWER_FRAC = 0.2; SOC_UPPER_FRAC = 0.8
EXTRA_GRID_COST = 4.0
MIN_ARBITRAGE_DIFF = 30
profit_cap = simulate_battery(df_2024, scenario="forecast")
results.append({"Scenario": "Capacity 4.44 MWh", "Profit (AUD)": profit_cap})

# Max Power = 2.22 MW (increase from 1.11)
BATTERY_CAPACITY_MWH = 2.22
BATTERY_MAX_POWER_MW = 2.22
ROUND_TRIP_EFF = 0.855; EFF_C = EFF_D = np.sqrt(ROUND_TRIP_EFF)
SOC_LOWER_FRAC = 0.2; SOC_UPPER_FRAC = 0.8
EXTRA_GRID_COST = 4.0
MIN_ARBITRAGE_DIFF = 30
profit_power = simulate_battery(df_2024, scenario="forecast")
results.append({"Scenario": "Max Power 2.22 MW", "Profit (AUD)": profit_power})

# Round-Trip Efficiency = 95% (increase from 85.5%)
BATTERY_CAPACITY_MWH = 2.22
BATTERY_MAX_POWER_MW = 1.11
ROUND_TRIP_EFF = 0.95; EFF_C = EFF_D = np.sqrt(ROUND_TRIP_EFF)
SOC_LOWER_FRAC = 0.2; SOC_UPPER_FRAC = 0.8
EXTRA_GRID_COST = 4.0
MIN_ARBITRAGE_DIFF = 30
profit_eff = simulate_battery(df_2024, scenario="forecast")
results.append({"Scenario": "Efficiency 95%", "Profit (AUD)": profit_eff})

# Rolling Window = 30 days (increase from 7 days)
BATTERY_CAPACITY_MWH = 2.22
BATTERY_MAX_POWER_MW = 1.11
ROUND_TRIP_EFF = 0.855; EFF_C = EFF_D = np.sqrt(ROUND_TRIP_EFF)
SOC_LOWER_FRAC = 0.2; SOC_UPPER_FRAC = 0.8
EXTRA_GRID_COST = 4.0
MIN_ARBITRAGE_DIFF = 30
# Recompute price thresholds with 30-day rolling window
window_size_30 = 30 * (24*60 // INTERVAL_MIN)
df_window30 = df_2024.copy()
df_window30["low_thr0"]  = df_window30["price"].shift(1).rolling(window_size_30, min_periods=1).quantile(0.2)
df_window30["low_thr"]   = df_window30["price"].shift(1).rolling(window_size_30, min_periods=1).quantile(0.1)
df_window30["high_thr0"] = df_window30["price"].shift(1).rolling(window_size_30, min_periods=1).quantile(0.8)
df_window30["high_thr"]  = df_window30["price"].shift(1).rolling(window_size_30, min_periods=1).quantile(0.9)
profit_window = simulate_battery(df_window30, scenario="forecast")
results.append({"Scenario": "Window 30d", "Profit (AUD)": profit_window})

# SOC Range = 10%–90% (wider from 20%–80%)
BATTERY_CAPACITY_MWH = 2.22
BATTERY_MAX_POWER_MW = 1.11
ROUND_TRIP_EFF = 0.855; EFF_C = EFF_D = np.sqrt(ROUND_TRIP_EFF)
SOC_LOWER_FRAC = 0.1; SOC_UPPER_FRAC = 0.9
EXTRA_GRID_COST = 4.0
MIN_ARBITRAGE_DIFF = 30
profit_soc = simulate_battery(df_2024, scenario="forecast")
results.append({"Scenario": "SOC 10-90%", "Profit (AUD)": profit_soc})

# Grid Charging Extra Cost = $0 (reduce from $4/MWh)
BATTERY_CAPACITY_MWH = 2.22
BATTERY_MAX_POWER_MW = 1.11
ROUND_TRIP_EFF = 0.855; EFF_C = EFF_D = np.sqrt(ROUND_TRIP_EFF)
SOC_LOWER_FRAC = 0.2; SOC_UPPER_FRAC = 0.8
EXTRA_GRID_COST = 0.0
MIN_ARBITRAGE_DIFF = 30
profit_grid = simulate_battery(df_2024, scenario="forecast")
results.append({"Scenario": "GridCost $0", "Profit (AUD)": profit_grid})

# Minimum Arbitrage Threshold = 10 AUD/MWh (decrease from 30)
BATTERY_CAPACITY_MWH = 2.22
BATTERY_MAX_POWER_MW = 1.11
ROUND_TRIP_EFF = 0.855; EFF_C = EFF_D = np.sqrt(ROUND_TRIP_EFF)
SOC_LOWER_FRAC = 0.2; SOC_UPPER_FRAC = 0.8
EXTRA_GRID_COST = 4.0
MIN_ARBITRAGE_DIFF = 10
profit_thr = simulate_battery(df_2024, scenario="forecast")
results.append({"Scenario": "Threshold 10 AUD", "Profit (AUD)": profit_thr})

# Convert results to DataFrame for display and CSV output
df_results = pd.DataFrame(results)
# 计算并添加 Relative Change (%) 列（相对Baseline利润的百分比变化）
df_results["Relative Change (%)"] = ((df_results["Profit (AUD)"] - profit_base) / profit_base * 100).round(1)
# 打印每个场景的利润和相对变化（Profit保留2位小数，Relative Change保留1位小数）
formatters = {"Profit (AUD)": "{:,.2f}".format,
              "Relative Change (%)": "{:.1f}".format}
print(df_results.to_string(index=False, formatters=formatters))
# 保存结果到 CSV 文件
df_results.to_csv("sensitivity_results_2024.csv", index=False)

# Plot bar chart of profits for each scenario
plt.figure(figsize=(12, 3))
bars = plt.bar(df_results["Scenario"], df_results["Profit (AUD)"], color="skyblue")
plt.title("2024 Profit Sensitivity Analysis", fontsize=12)
plt.xlabel("Parameter Scenario", fontsize=11)
plt.ylabel("Profit in 2024 (AUD)", fontsize=11)
# Add value labels on each bar (profit and relative change)
max_profit = abs(df_results["Profit (AUD)"]).max()
for i, bar in enumerate(bars):
    y_val = bar.get_height()
    profit_label = f"${y_val:,.0f}"
    rc_val = df_results["Relative Change (%)"].iloc[i]
    if rc_val > 0:
        rc_label = f"+{rc_val:.1f}%"
    else:
        rc_label = f"{rc_val:.1f}%"
    plt.text(bar.get_x() + bar.get_width()/2,
             y_val + 0.01 * max_profit,
             f"{profit_label}\n{rc_label}",
             ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=45, ha="right")
plt.ylim(bottom=200000)
plt.tight_layout()
# Save the figure as an image file
plt.savefig("parameter_sensitivity_profit_2024.png")
plt.show()
