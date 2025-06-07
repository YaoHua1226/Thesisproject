import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.dates as mdates

# 1. Load 2023–2024 5-minute QLD1 price data for training
train_frames = []
for year in [2023, 2024]:
    for month in range(1, 13):
        path = f'PRICE_AND_DEMAND_{year}{month:02d}_QLD1.csv'
        try:
            dfm = pd.read_csv(path, usecols=['SETTLEMENTDATE', 'RRP'])
        except FileNotFoundError:
            continue
        dfm['SETTLEMENTDATE'] = pd.to_datetime(dfm['SETTLEMENTDATE'])
        dfm.sort_values('SETTLEMENTDATE', inplace=True)
        train_frames.append(dfm)
training_df = pd.concat(train_frames, ignore_index=True)
training_df.sort_values('SETTLEMENTDATE', inplace=True)
training_df.reset_index(drop=True, inplace=True)

# 2. Generate Fourier features for daily & weekly seasonality
def fourier_terms(datetimes, K_day, K_week):
    period_day = 288  # 24h * (60/5)
    minutes = datetimes.dt.hour * 60 + datetimes.dt.minute
    idx_day = (minutes // 5).values
    n = len(datetimes)
    # daily harmonics
    Xd = np.zeros((n, 2*K_day))
    for k in range(1, K_day+1):
        Xd[:,2*(k-1)]   = np.sin(2*np.pi*k*idx_day/period_day)
        Xd[:,2*(k-1)+1] = np.cos(2*np.pi*k*idx_day/period_day)
    # weekly harmonics
    period_week = period_day*7
    dow = datetimes.dt.dayofweek.values
    idx_week = dow*period_day + idx_day
    Xw = np.zeros((n, 2*K_week))
    for k in range(1, K_week+1):
        Xw[:,2*(k-1)]   = np.sin(2*np.pi*k*idx_week/period_week)
        Xw[:,2*(k-1)+1] = np.cos(2*np.pi*k*idx_week/period_week)
    return np.hstack([Xd, Xw])

# set Fourier orders
K_day, K_week = 12, 4
X_fourier = fourier_terms(training_df['SETTLEMENTDATE'], K_day, K_week)

# 3. Fit ARIMAX via OLS (AR(7) + Fourier exogenous terms)
p = 7
# create lag features
for i in range(1, p+1):
    training_df[f'lag{i}'] = training_df['RRP'].shift(i)
training_df.dropna(inplace=True)

y = training_df['RRP'].values
lags = training_df[[f'lag{i}' for i in range(1,p+1)]].values
Xf = X_fourier[p:,:]  # align with dropped rows
# stack lags + Fourier + intercept
X_full = np.hstack([lags, Xf, np.ones((len(y),1))])
coef, *_ = np.linalg.lstsq(X_full, y, rcond=None)
phi        = coef[:p]
beta_four = coef[p:-1]
intercept  = coef[-1]

# 4. Forecast 2025 at 5-min intervals
forecast_index = pd.date_range('2025-01-01 00:05','2025-12-31 23:55',freq='5min')
Xf_fore = fourier_terms(pd.Series(forecast_index), K_day, K_week)
y_pred = np.zeros(len(forecast_index))
# seed with last p actuals from training
buffer = training_df['RRP'].iloc[-p:].values.copy()

for t in range(len(forecast_index)):
    # AR part + Fourier part + intercept
    y_t = intercept + phi.dot(buffer[::-1]) + beta_four.dot(Xf_fore[t])
    y_pred[t] = y_t
    buffer = np.roll(buffer, -1)
    buffer[-1] = y_t

# --- SAVE FULL-YEAR FORECAST TO CSV ---
df_forecast = pd.DataFrame({
    'SETTLEMENTDATE': forecast_index,
    'RRP_pred': y_pred
})
df_forecast.to_csv('2025_full_year_forecast.csv', index=False)
print("Exported 2025_full_year_forecast.csv")

# 5. Evaluate Jan–Apr 2025 with MAE & RMSE
# load actuals
actuals = []
for ym in ['202501','202502','202503','202504']:
    df = pd.read_csv(f'PRICE_AND_DEMAND_{ym}_QLD1.csv', usecols=['SETTLEMENTDATE','RRP'])
    df['SETTLEMENTDATE']=pd.to_datetime(df['SETTLEMENTDATE'])
    df.sort_values('SETTLEMENTDATE',inplace=True)
    actuals.append(df)
df_act = pd.concat(actuals,ignore_index=True).set_index('SETTLEMENTDATE')
series_pred = pd.Series(y_pred, index=forecast_index)
series_act  = df_act['RRP'].reindex(series_pred.index)

metrics = {}
for month in ['2025-01','2025-02','2025-03','2025-04']:
    mask = series_pred.index.strftime('%Y-%m')==month
    y_t = series_act[mask].values
    y_p = series_pred[mask].values
    mae  = np.nanmean(np.abs(y_t-y_p))
    rmse = np.sqrt(np.nanmean((y_t-y_p)**2))
    metrics[month] = (mae, rmse)

print("Monthly MAE & RMSE Jan–Apr 2025:")
for m,(mae,rmse) in metrics.items():
    print(f"{m}: MAE={mae:.2f}, RMSE={rmse:.2f}")

# 6. Plot full-year forecast
plt.figure(figsize=(12,5))
plt.plot(series_pred.index, series_pred.values, color='steelblue')
plt.title('2025 Annual Electricity Price Forecast (5-min)')
plt.xlabel('Date'); plt.ylabel('Price ($/MWh)')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()
plt.savefig('2025_full_year_forecast.png')
plt.close()

# 7. Plot Jan–Apr actual vs forecast with broken axis for spikes
threshold = 1000
for month in ['2025-01','2025-02','2025-03','2025-04']:
    mask = series_pred.index.strftime('%Y-%m')==month
    t = series_pred.index[mask]
    a = series_act[mask].values
    p = series_pred[mask].values
    label = pd.to_datetime(month+'-01').strftime('%Y %b')
    if np.nanmax(a)>threshold:
        fig, (top,bot) = plt.subplots(2,1,sharex=True,figsize=(10,5),
                                      gridspec_kw={'height_ratios':[1,3]})
        bot.plot(t, np.where(a>threshold,np.nan,a), color='orange', label='Actual')
        bot.plot(t, np.where(p>threshold,np.nan,p), color='steelblue', label='Forecast')
        bot.set_ylabel('Price')
        bot.legend(loc='upper left')
        top.plot(t, np.where(a>threshold,a,np.nan),'o',color='orange')
        top.plot(t, np.where(p>threshold,p,np.nan),'o',color='steelblue')
        top.set_ylabel('Price')
        top.spines['bottom'].set_visible(False)
        bot.spines['top'].set_visible(False)
        top.tick_params(labelbottom=False)
        d=.005;kwargs=dict(color='k',clip_on=False)
        top.plot((-d,d),(-d,d),transform=top.transAxes,**kwargs)
        top.plot((1-d,1+d),(-d,d),transform=top.transAxes,**kwargs)
        kwargs.update(transform=bot.transAxes)
        bot.plot((-d,d),(1-d,1+d),**kwargs); bot.plot((1-d,1+d),(1-d,1+d),**kwargs)
        top.set_title(f'{label} Actual vs Forecast')
        bot.set_xlabel('Date')
        plt.tight_layout()
        plt.show()
        plt.savefig(f'{month.replace("-","")}_actual_vs_pred.png')
        plt.close(fig)
    else:
        plt.figure(figsize=(10,5))
        plt.plot(t,a,color='orange',label='Actual')
        plt.plot(t,p,color='steelblue',label='Forecast')
        plt.title(f'{label} Actual vs Forecast')
        plt.xlabel('Date'); plt.ylabel('Price ($/MWh)')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
        plt.savefig(f'{month.replace("-","")}_actual_vs_pred.png')
        plt.close()

# --- 8. 2024 Full-Year Forecast vs Actual ---

# 8.1 Load only 2023 data for re-training
train_2023 = []
for month in range(1, 13):
    path = f'PRICE_AND_DEMAND_2023{month:02d}_QLD1.csv'
    df = pd.read_csv(path, usecols=['SETTLEMENTDATE','RRP'])
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
    df.sort_values('SETTLEMENTDATE', inplace=True)
    train_2023.append(df)
df23 = pd.concat(train_2023, ignore_index=True)
df23.sort_values('SETTLEMENTDATE', inplace=True)
df23.reset_index(drop=True, inplace=True)

# 8.2 Generate Fourier features for 2023 training
X23_fourier = fourier_terms(df23['SETTLEMENTDATE'], K_day, K_week)

# 8.3 Build AR(7)+Fourier via OLS on 2023
p = 7
for i in range(1, p+1):
    df23[f'lag{i}'] = df23['RRP'].shift(i)
df23.dropna(inplace=True)
y23 = df23['RRP'].values
lags23 = df23[[f'lag{i}' for i in range(1,p+1)]].values
Xf23 = X23_fourier[p:,:]
X_full23 = np.hstack([lags23, Xf23, np.ones((len(y23),1))])
coef23, *_ = np.linalg.lstsq(X_full23, y23, rcond=None)
phi23       = coef23[:p]
beta_f23    = coef23[p:-1]
intercept23 = coef23[-1]

# 8.4 Forecast entire 2024 at 5-minute intervals
idx24 = pd.date_range('2024-01-01 00:05','2024-12-31 23:55',freq='5min')
Xf24 = fourier_terms(pd.Series(idx24), K_day, K_week)
y24 = np.zeros(len(idx24))
buf = df23['RRP'].iloc[-p:].values.copy()  # seed with last p points of 2023

for t in range(len(idx24)):
    y_t = intercept23 + phi23.dot(buf[::-1]) + beta_f23.dot(Xf24[t])
    y24[t] = y_t
    buf = np.roll(buf, -1)
    buf[-1] = y_t

# ——— 在此处添加：导出 2024 年全年的预测到 CSV ———
df_forecast24 = pd.DataFrame({
    'SETTLEMENTDATE': idx24,
    'RRP_pred':    y24
})
df_forecast24.to_csv('2024_full_year_forecast.csv', index=False)
print("Exported 2024_full_year_forecast.csv")

# 8.5 Load actual 2024 data
# …后续绘图和评估代码…

# 8.5 Load actual 2024 data
actual24 = []
for month in range(1, 13):
    path = f'PRICE_AND_DEMAND_2024{month:02d}_QLD1.csv'
    df = pd.read_csv(path, usecols=['SETTLEMENTDATE','RRP'])
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
    df.sort_values('SETTLEMENTDATE', inplace=True)
    actual24.append(df)
df_act24 = pd.concat(actual24, ignore_index=True).set_index('SETTLEMENTDATE')
ser_pred24 = pd.Series(y24, index=idx24)
ser_act24  = df_act24['RRP'].reindex(idx24)

# 8.6 Plot full-year 2024 comparison
plt.figure(figsize=(12,5))
plt.plot(ser_act24.index, ser_act24.values, label='Actual 2024', color='orange', alpha=0.6)
plt.plot(ser_pred24.index, ser_pred24.values, label='Forecast 2024', color='steelblue', alpha=0.6)
plt.title('2024 Full-Year Actual vs Forecast (5-minute)')
plt.xlabel('Date'); plt.ylabel('Price ($/MWh)')
plt.legend(loc='upper right')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig('2024_full_year_actual_vs_forecast.png')
plt.close()

print("Saved plot: 2024_full_year_actual_vs_forecast.png")

