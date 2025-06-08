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
    period_day = 288  # 24h * (60/5) for 5-min intervals in a day
    minutes = datetimes.dt.hour * 60 + datetimes.dt.minute
    idx_day = (minutes // 5).values
    n = len(datetimes)
    # daily harmonics
    Xd = np.zeros((n, 2*K_day))
    for k in range(1, K_day+1):
        Xd[:, 2*(k-1)]     = np.sin(2*np.pi*k * idx_day / period_day)
        Xd[:, 2*(k-1) + 1] = np.cos(2*np.pi*k * idx_day / period_day)
    # weekly harmonics
    period_week = period_day * 7
    dow = datetimes.dt.dayofweek.values
    idx_week = dow * period_day + idx_day
    Xw = np.zeros((n, 2*K_week))
    for k in range(1, K_week+1):
        Xw[:, 2*(k-1)]     = np.sin(2*np.pi*k * idx_week / period_week)
        Xw[:, 2*(k-1) + 1] = np.cos(2*np.pi*k * idx_week / period_week)
    return np.hstack([Xd, Xw])

# set Fourier orders (number of harmonics for day and week)
K_day, K_week = 12, 4
X_fourier = fourier_terms(training_df['SETTLEMENTDATE'], K_day, K_week)

# 3. Fit ARIMAX via OLS (AR(7) + Fourier exogenous terms)
p = 7
# create lag features for AR(p)
for i in range(1, p+1):
    training_df[f'lag{i}'] = training_df['RRP'].shift(i)
training_df.dropna(inplace=True)  # drop initial rows with NaN lags

y = training_df['RRP'].values
lags = training_df[[f'lag{i}' for i in range(1, p+1)]].values
Xf = X_fourier[p:, :]  # align Fourier matrix with dropped rows
# stack [lags, Fourier terms, intercept] into one design matrix
X_full = np.hstack([lags, Xf, np.ones((len(y), 1))])
# solve least squares: minimize ||X_full * coef - y||^2
coef, *_ = np.linalg.lstsq(X_full, y, rcond=None)
phi        = coef[:p]       # AR coefficients (length p)
beta_four  = coef[p:-1]     # Fourier coefficients
intercept  = coef[-1]       # intercept term

# 4. Forecast 2025–2033 at 5-min intervals and save outputs
forecast_years = range(2024, 2034)
# seed buffer with last p actual values from end of 2024
buffer = training_df['RRP'].iloc[-p:].values.copy()

for year in forecast_years:
    # create full-year datetime index at 5-min frequency for the forecast year
    forecast_index = pd.date_range(f'{year}-01-01 00:05', f'{year}-12-31 23:55', freq='5min')
    # generate Fourier terms for the forecast year's timestamps
    Xf_fore = fourier_terms(pd.Series(forecast_index), K_day, K_week)
    # array to hold predictions for this year
    y_pred = np.zeros(len(forecast_index))
    # iterative forecasting for each timestep
    for t in range(len(forecast_index)):
        # AR(7) part + seasonal Fourier part + intercept
        y_t = intercept + phi.dot(buffer[::-1]) + beta_four.dot(Xf_fore[t])
        y_pred[t] = y_t
        # update buffer: drop oldest value, append new prediction
        buffer = np.roll(buffer, -1)
        buffer[-1] = y_t
    # save predictions to CSV
    df_forecast = pd.DataFrame({
        'SETTLEMENTDATE': forecast_index,
        'RRP_pred': y_pred
    })
    df_forecast.to_csv(f'{year}_full_year_forecast.csv', index=False)
    print(f"Exported {year}_full_year_forecast.csv")
    # plot full-year forecast and save figure
    plt.figure(figsize=(12, 5))
    plt.plot(forecast_index, y_pred, color='steelblue')
    plt.title(f'{year} Annual Electricity Price Forecast (5-min)')
    plt.xlabel('Date'); plt.ylabel('Price ($/MWh)')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{year}_full_year_forecast.png')
    plt.show()
    plt.close()
    print(f"Saved {year}_full_year_forecast.png")
