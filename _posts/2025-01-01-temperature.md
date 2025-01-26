---
layout: splash
permalink: /temperature/
title: "Temperature Forecasting using SARIMA"
header:
  overlay_image: /assets/images/temperature/temperature.jpeg
excerpt: "Forecast the average monthly temperature in Rome using seasonal ARIMA methods."
---

```python
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
```


```python
cities = pd.read_csv('./GlobalLandTemperaturesByMajorCity.csv')
df = cities.loc[cities['City'] == 'Rome', ['dt','AverageTemperature']]
df.columns = ['date', 'y']
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.strftime('%b')
df.reset_index(drop=True, inplace=True)
df.set_index('date', inplace=True)
df = df.loc['1900':'2012']
df = df.asfreq('M', method='bfill')
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>year</th>
      <th>month</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-07-31</th>
      <td>24.731</td>
      <td>2012</td>
      <td>Aug</td>
    </tr>
    <tr>
      <th>2012-08-31</th>
      <td>18.922</td>
      <td>2012</td>
      <td>Sep</td>
    </tr>
    <tr>
      <th>2012-09-30</th>
      <td>14.501</td>
      <td>2012</td>
      <td>Oct</td>
    </tr>
    <tr>
      <th>2012-10-31</th>
      <td>10.528</td>
      <td>2012</td>
      <td>Nov</td>
    </tr>
    <tr>
      <th>2012-11-30</th>
      <td>4.150</td>
      <td>2012</td>
      <td>Dec</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,5))
sns.lineplot(data=df.reset_index(), x='date', y='y', label='Average Monthly Temp')
plt.title('Average Monthly Temperature in Rome from 1900 until 2012')
plt.xlabel('Year')
plt.ylabel("Temperature (°C)")
plt.legend();
```




    <matplotlib.legend.Legend at 0x16713e210>




    
![png](/assets/images/temperature/temperature-1.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 5))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec']
sns.boxplot(data=df, x='month', y='y', order=months, ax=ax)
sns.stripplot(data=df, x='month', y='y', ax=ax)
ax.set_xlabel('Month of the Year')
ax.set_ylabel('Temperature (°C)');
```


    
![png](/assets/images/temperature/temperature-2.png)
    



```python
year_avg = pd.pivot_table(df, values='y', index='year', aggfunc='mean')
year_avg['y_ma_10y'] = year_avg['y'].rolling(10).mean()
year_avg = year_avg.reset_index()
fig, ax = plt.subplots(figsize=(15, 5))
sns.lineplot(data=df, x='year', y='y', label='Monthly Temperature', ax=ax, ci=None)
sns.lineplot(data=year_avg, x='year', y='y_ma_10y', label='10Y mean', ax=ax, ci=None)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend();
```


    
![png](/assets/images/temperature/temperature-3.png)
    



```python
from statsmodels.tsa.seasonal import seasonal_decompose, STL
decomposition = STL(df.y, period=12).fit()

fig, (ax0, ax1, ax2, ax3) = plt.subplots(figsize=(10, 8), nrows=4, ncols=1, sharex=True)
ax0.plot(decomposition.observed)
ax0.set_ylabel('Observed')
ax1.plot(decomposition.trend)
ax1.set_ylabel('Trend')
ax2.plot(decomposition.seasonal)
ax2.set_ylabel('Seasonal')
ax3.plot(decomposition.resid)
ax3.set_ylabel('Residuals')
fig.tight_layout()
```


    
![png](/assets/images/temperature/temperature-4.png)
    



```python
from statsmodels.tsa.stattools import adfuller
res = adfuller(df.y)
print(f'Observed data: ADF statistics: {res[0]}, p-value: {res[1]}')
res = adfuller(np.diff(df.y, n=1))
print(f'First-order differenced data: ADF statistics: {res[0]}, p-value: {res[1]}')
res = adfuller(np.diff(df.y, n=12))
print(f'First-order differenced data: ADF statistics: {res[0]}, p-value: {res[1]}')
```

    Observed data: ADF statistics: -3.4263718902503433, p-value: 0.010092042879276453
    First-order differenced data: ADF statistics: -16.35212390756837, p-value: 2.925519649193536e-29
    First-order differenced data: ADF statistics: -25.706176135107768, p-value: 0.0



```python
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
```


```python
def optimize_SARIMAX(endog, exog, orders, d, D, s, trend):
    import warnings
    results = []
    for order in tqdm(orders):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = SARIMAX(
                    endog,
                    exog,
                    order=(order[0], d, order[1]),
                    seasonal_order=(order[2], D, order[3], s),
                    simple_differencing=False,
                    trend=trend,
                ).fit(disp=False)
        except:
            continue

        results.append([*order, model.aic, model.bic])
    
    results = pd.DataFrame(results)
    results.columns = ['p', 'q', 'P', 'Q', 'AIC', 'BIC']
    results = results.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return results
```


```python
df_train = df.y[:-120]
df_test = df.y[-120:]
```


```python
from itertools import product
orders = list(product(
    range(0, 4),
    range(0, 4),
    range(0, 4),
    range(0, 4),
))
d, D, s = 1, 1, 12
results = optimize_SARIMAX(df_train, None, orders=orders, d=d, D=D, s=s, trend='c')
```

    100%|██████████| 256/256 [40:13<00:00,  9.43s/it]  



```python
results = results.set_index(['p', 'q', 'P', 'Q'])
```


```python
p, q, P, Q = results.index.values[0]
print(f"Selected p={p}, q={q}, P={P}, Q={Q}")
```

    Selected p=1, q=1, P=3, Q=2



```python

```


```python
model = SARIMAX(df_train, order=(p, d, q), seasonal_order=(P, D, Q, s), simple_differencing=False)
fitted_model = model.fit(disp=False)
fitted_model.plot_diagnostics(figsize=(10, 8));
```


    
![png](/assets/images/temperature/temperature-5.png)
    



```python
residuals = fitted_model.resid
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(residuals, np.arange(1, 11, 11))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lb_stat</th>
      <th>lb_pvalue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2.900733</td>
      <td>0.088539</td>
    </tr>
  </tbody>
</table>
</div>




```python
def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, window: int, method: str) -> list:
    from tqdm import trange
    total_len = train_len + horizon
    if method == 'last_season':
        pred_last_season = []
        for i in range(train_len, total_len, window):
            last_season = df['y'][i - window:i].values
            pred_last_season.extend(last_season)
        return pred_last_season

    elif method == 'SARIMA':
        pred_SARIMA = []
        for i in trange(train_len, total_len, window):
            model = SARIMAX(df['y'][:i], order=(2, 1, 3), seasonal_order=(1, 1, 3, 12),
                            simple_differencing=False)
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_SARIMA.extend(oos_pred)
        return pred_SARIMA
```


```python
TRAIN_LEN = len(df_train)
HORIZON = len(df_test)
WINDOW = 12

last_season = rolling_forecast(df, TRAIN_LEN, HORIZON, WINDOW, 'last_season')
sarima = rolling_forecast(df, TRAIN_LEN, HORIZON, WINDOW, 'SARIMA')
```

    /Users/marzio/envs/base/lib/python3.12/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    /Users/marzio/envs/base/lib/python3.12/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    /Users/marzio/envs/base/lib/python3.12/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    /Users/marzio/envs/base/lib/python3.12/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    /Users/marzio/envs/base/lib/python3.12/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    /Users/marzio/envs/base/lib/python3.12/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    /Users/marzio/envs/base/lib/python3.12/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    /Users/marzio/envs/base/lib/python3.12/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    /Users/marzio/envs/base/lib/python3.12/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    /Users/marzio/envs/base/lib/python3.12/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "



```python
def compute_rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_true, y_pred))

naive = df_test.shift(1).dropna()

print(f"RMSE:")
print(f'- naïve:       {compute_rmse(df_test.iloc[1:], naive):.4f} (°C)')
print(f'- last season: {compute_rmse(df_test, last_season):.4f} (°C)')
print(f'- SARIMA:      {compute_rmse(df_test, sarima):.4f} (°C)')
```

    RMSE:
    - naïve:       3.9803 (°C)
    - last season: 1.8120 (°C)
    - SARIMA:      1.2857 (°C)



```python
def compute_mae(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)

naive = df_test.shift(1).dropna()

print(f"MAE:")
print(f'- naïve:       {compute_mae(df_test.iloc[1:], naive):.4f} (°C)')
print(f'- last season: {compute_mae(df_test, last_season):.4f} (°C)')
print(f'- SARIMA:      {compute_mae(df_test, sarima):.4f} (°C)')
```

    MAE:
    - naïve:       3.5163 (°C)
    - last season: 1.4551 (°C)
    - SARIMA:      0.9814 (°C)



```python
fig, ax = plt.subplots(figsize=(15, 5))
plt.plot(df_test.index, df_test.values, label='Actual', linewidth=4, alpha=0.75)
plt.plot(df_test.index, last_season, '-s', label='Last season')
plt.plot(df_test.index, sarima, '-o', label='SARIMA')
plt.xlabel('Date')
plt.ylabel("Temperature (°C)")
plt.legend();
```


    
![png](/assets/images/temperature/temperature-6.png)
    

