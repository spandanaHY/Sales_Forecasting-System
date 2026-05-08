import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle, os, warnings
warnings.filterwarnings('ignore')

os.makedirs('static/graphs', exist_ok=True)
os.makedirs('datasets', exist_ok=True)

print("=" * 55)
print("   SALES FORECASTING — TRAINING")
print("=" * 55)

# STEP 1: LOAD with encoding fix
print("\n[1/7] Loading Dataset...")
df = None
for enc in ['latin-1', 'cp1252', 'utf-8', 'iso-8859-1']:
    try:
        df = pd.read_csv('datasets/sales_data.csv', encoding=enc)
        print(f"     Loaded OK with encoding: {enc}")
        break
    except Exception as e:
        continue

if df is None:
    print("ERROR: Could not load file!")
    exit()

print(f"     Rows: {len(df)} | Columns: {list(df.columns)}")

# STEP 2: Auto-detect date and sales columns
print("\n[2/7] Detecting columns...")
date_col = None
for col in df.columns:
    if 'date' in col.lower():
        date_col = col
        break
if not date_col:
    for col in df.columns:
        try:
            pd.to_datetime(df[col].dropna().iloc[:5])
            date_col = col
            break
        except:
            continue

sales_col = None
for col in df.columns:
    if any(x in col.lower() for x in ['sales','revenue','amount','price','units_sold']):
        sales_col = col
        break
if not sales_col:
    for col in df.columns:
        if df[col].dtype in ['float64','int64']:
            sales_col = col
            break

print(f"     Date column  : {date_col}")
print(f"     Sales column : {sales_col}")

# STEP 3: Preprocess
print("\n[3/7] Preprocessing...")
df[date_col]  = pd.to_datetime(df[date_col], dayfirst=False, errors='coerce')
df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
df = df.dropna(subset=[date_col, sales_col])
df = df.sort_values(date_col)

daily = df.groupby(date_col)[sales_col].sum().reset_index()
daily.columns = ['Date','Revenue']
daily = daily.set_index('Date').asfreq('D', method='ffill')
daily['Revenue'] = daily['Revenue'].fillna(method='ffill')

print(f"     Range: {daily.index.min().date()} to {daily.index.max().date()}")
print(f"     Days : {len(daily)}")
print(f"     Avg  : {daily['Revenue'].mean():,.2f}")

# STEP 4: EDA Graphs
print("\n[4/7] Generating EDA Graphs...")

plt.figure(figsize=(14,5))
plt.plot(daily.index, daily['Revenue'], color='#00e676', linewidth=0.9)
plt.fill_between(daily.index, daily['Revenue'], alpha=0.12, color='#00e676')
plt.title('Daily Revenue Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Date'); plt.ylabel('Revenue')
plt.tight_layout()
plt.savefig('static/graphs/sales_trend.png', dpi=100, bbox_inches='tight')
plt.close()
print("     sales_trend.png done")

monthly = daily.resample('M').sum()
plt.figure(figsize=(14,5))
plt.bar(monthly.index, monthly['Revenue'], color='#00bcd4', width=20)
plt.title('Monthly Total Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Month'); plt.ylabel('Revenue')
plt.tight_layout()
plt.savefig('static/graphs/monthly_revenue.png', dpi=100, bbox_inches='tight')
plt.close()
print("     monthly_revenue.png done")

try:
    period = 365 if len(daily) >= 730 else 30
    decomp = seasonal_decompose(daily['Revenue'], model='additive', period=period)
    fig, axes = plt.subplots(4,1,figsize=(14,10))
    items = [('Observed',decomp.observed,'#00e676'),
             ('Trend',decomp.trend,'#00bcd4'),
             ('Seasonality',decomp.seasonal,'#ff6f00'),
             ('Residual',decomp.resid,'#ef5350')]
    for ax,(title,data,color) in zip(axes,items):
        ax.plot(data, color=color, linewidth=0.9)
        ax.set_title(title)
    plt.suptitle('Time Series Decomposition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/graphs/decomposition.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("     decomposition.png done")
except Exception as e:
    print(f"     decomposition skipped: {e}")

# STEP 5: Train ARIMA
print("\n[5/7] Training ARIMA(5,1,0)...")
use_days   = min(365, len(daily))
train_data = daily['Revenue'][-use_days:]
split      = int(len(train_data) * 0.8)
train      = train_data[:split]
test       = train_data[split:]
print(f"     Train={len(train)} | Test={len(test)}")
print("     Please wait...")

model  = ARIMA(train, order=(5,1,0))
fitted = model.fit()
print("     ARIMA trained!")

# STEP 6: Evaluate
print("\n[6/7] Evaluating...")
preds  = np.array(fitted.forecast(steps=len(test))).clip(0)
actual = test.values

mae  = mean_absolute_error(actual, preds)
mse  = mean_squared_error(actual, preds)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actual-preds)/np.where(actual==0,1,actual)))*100

print(f"     MAE={mae:.2f} | MSE={mse:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}%")
metrics = {'MAE':round(mae,2),'MSE':round(mse,2),'RMSE':round(rmse,2),'MAPE':round(mape,2)}
pickle.dump(metrics, open('arima_metrics.pkl','wb'))

plt.figure(figsize=(14,5))
plt.plot(test.index, actual, label='Actual',    color='#00e676', linewidth=1.5)
plt.plot(test.index, preds,  label='Predicted', color='#ff6f00', linewidth=1.5, linestyle='--')
plt.fill_between(test.index, actual, preds, alpha=0.1, color='#ef5350')
plt.title('ARIMA — Actual vs Predicted', fontsize=14, fontweight='bold')
plt.xlabel('Date'); plt.ylabel('Revenue')
plt.legend(); plt.tight_layout()
plt.savefig('static/graphs/actual_vs_predicted.png', dpi=100, bbox_inches='tight')
plt.close()
print("     actual_vs_predicted.png done")

plt.figure(figsize=(7,4))
bars = plt.bar(['MAE','RMSE'],[mae,rmse], color=['#00e676','#00bcd4'], width=0.4)
for bar,val in zip(bars,[mae,rmse]):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
plt.title('Error Metrics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('static/graphs/error_metrics.png', dpi=100, bbox_inches='tight')
plt.close()
print("     error_metrics.png done")

use_future   = min(180, len(daily))
future_model = ARIMA(daily['Revenue'][-use_future:], order=(5,1,0))
future_fit   = future_model.fit()
forecast     = np.array(future_fit.forecast(steps=30)).clip(0)
last_date    = daily.index[-1]
future_dates = pd.date_range(start=last_date+pd.Timedelta(days=1), periods=30)

plt.figure(figsize=(14,5))
hist = min(90, len(daily))
plt.plot(daily.index[-hist:], daily['Revenue'][-hist:],
         label='Historical', color='#00e676', linewidth=1.5)
plt.plot(future_dates, forecast, label='30-Day Forecast',
         color='#00bcd4', linewidth=2, linestyle='--')
plt.fill_between(future_dates, forecast*0.92, forecast*1.08,
                 alpha=0.2, color='#00bcd4', label='Confidence Band')
plt.axvline(x=last_date, color='#ff6f00', linestyle=':', linewidth=1.5)
plt.title('30-Day Sales Forecast (ARIMA)', fontsize=14, fontweight='bold')
plt.xlabel('Date'); plt.ylabel('Revenue')
plt.legend(); plt.tight_layout()
plt.savefig('static/graphs/forecast_future.png', dpi=100, bbox_inches='tight')
plt.close()
print("     forecast_future.png done")

# STEP 7: Save
print("\n[7/7] Saving model files...")
pickle.dump(fitted,     open('arima_model.pkl',  'wb'))
pickle.dump(daily,      open('daily_data.pkl',   'wb'))
pickle.dump(future_fit, open('arima_future.pkl', 'wb'))

print("\n" + "=" * 55)
print(f"MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}%")
print("=" * 55)
