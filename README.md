# 📈 Mastering Time Series Analysis

_A comprehensive guide for Data Scientists & Data Analysts_        
**Author:** Abdelrahman Ahmed Ezzat        

---

## 📚 Table of Contents    

1. [Introduction](#introduction)      
2. [Time Series Decomposition](#time-series-decomposition)    
3. [Handling Outliers](#handling-outliers)  
4. [Stationarity](#stationarity)  
5. [Forecasting Models](#forecasting-models)
6. [Smoothing Methods](#smoothing-methods)
7. [ACF & PACF](#acf--pacf)
8. [Data Preprocessing](#data-preprocessing)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Conclusion & Next Steps](#conclusion--next-steps)

---

## 📌 Introduction

**What is a Time Series?**

> A sequence of data points indexed in time order, typically collected at equal intervals.    

**Examples:**  
- Stock prices & volumes              
- Sales & demand forecasting        
- IoT/sensor data        
- Web & app analytics      
- Healthcare data      

**Key Python Setup:**

```python
!pip install yfinance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Download Apple stock data
stock_data = yf.download('AAPL', start='2024-01-01')
stock_data.head()

# Plot
plt.figure(figsize=(10,5))
plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='#FF9140')
plt.title('AAPL Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
````

---

## 🔍 Time Series Decomposition

**Components:**

* **Trend (T)**: Long-term direction
* **Seasonality (S)**: Regular repeating patterns
* **Cyclic (C)**: Irregular, long-period fluctuations
* **Residual (R)**: Random noise

**Models:**

* **Additive:** `Y = T + S + C + R`
* **Multiplicative:** `Y = T × S × C × R`

**STL Decomposition Example:**

```python
from statsmodels.tsa.seasonal import STL

stl_result = STL(stock_data['Close'], period=30).fit()
plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(stl_result.observed, color='#FF914D')
plt.title('STL Decomposition of Stock Prices')

plt.subplot(4, 1, 2)
plt.plot(stl_result.trend, color='#FF914D')
plt.ylabel('Trend')

plt.subplot(4, 1, 3)
plt.plot(stl_result.seasonal, color='#FF914D')
plt.ylabel('Seasonal')

plt.subplot(4, 1, 4)
plt.plot(stl_result.resid, color='#FF914D')
plt.ylabel('Residual')

plt.tight_layout()
plt.show()
```

---

## ⚡ Handling Outliers

**Impact:** Outliers distort trends & forecasts.
**Detection Methods:**

* **Statistical:** Z-score, IQR
* **Rolling:** Rolling median, MAD
* **ML:** Isolation Forest, One-Class SVM, DBSCAN

**Example:**

```python
from scipy import stats

z_scores = stats.zscore(stock_data['Close'].dropna())
threshold = 3
outliers = stock_data['Close'][np.abs(z_scores) > threshold]
print(outliers)
```

---

## 🔑 Stationarity

Many forecasting models assume stationarity.

**Tests:**

* **ADF Test**
* **KPSS Test**

```python
from statsmodels.tsa.stattools import adfuller, kpss

adf_result = adfuller(stock_data['Close'].dropna())
print(f'ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}')
```

---

## 📈 Forecasting Models

**Univariate:** AR, MA, ARMA, ARIMA, SARIMA      
**Multivariate:** VAR, VARMA, VARIMA        

---

## 🎛️ Smoothing Methods

**Types:**

* SMA (Simple Moving Average)
* WMA (Weighted)
* EMA (Exponential)

```python
window = 5
stock_data['SMA'] = stock_data['Close'].rolling(window=window).mean()
```

---

## 📉 ACF & PACF

* **ACF:** Correlation with lagged values.
* **PACF:** Direct correlation with lagged values, removing intermediate lags.

Use to identify `p` and `q` for ARIMA.

---

## 🔍 Data Preprocessing

* Handle **missing values**
* Make data **stationary** (differencing, transformations)
* Resample (downsample/upsample)

```python
# Example: Linear interpolation
stock_data['Close'] = stock_data['Close'].interpolate()
```

---

## 📊 Evaluation Metrics

* MAE
* MSE
* RMSE
* MAPE
* AIC/BIC

---

## ✅ Conclusion & Next Steps

**Key Takeaways:**

* Decompose series to understand patterns.
* Test for stationarity.
* Choose models that match data characteristics.
* Use proper preprocessing & evaluation metrics.

**Next:**

* Explore advanced models (Prophet, TBATS, LSTM)
* Anomaly detection
* Multivariate time series
* Real-time analytics

---

## 👋 Stay Curious!

*This repo is a living guide — feel free to contribute or fork!*
