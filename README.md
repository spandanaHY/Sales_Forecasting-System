#  Sales Forecasting System using ARIMA

##  Project Description

This project analyzes historical sales data and predicts future sales trends using the ARIMA time series forecasting model.
The system performs preprocessing, exploratory data analysis (EDA), model training, forecasting, and visualization of future sales predictions.

---

##  Features

* Automatic dataset loading with encoding handling
* Automatic date and sales column detection
* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Time series decomposition
* Sales forecasting using ARIMA
* Future 30-day revenue prediction
* Graph generation and model evaluation

---

##  Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Statsmodels
* Scikit-learn
* Pickle

---

##  Project Structure

```text id="z5qtx4"
Sales-Forecasting/
│
├── train_model.py
├── arima_model.pkl
├── arima_future.pkl
├── arima_metrics.pkl
├── daily_data.pkl
│
├── datasets/
│   └── sales_data.csv
│
├── static/
│   └── graphs/
│       ├── sales_trend.png
│       ├── monthly_revenue.png
│       ├── decomposition.png
│       ├── actual_vs_predicted.png
│       ├── error_metrics.png
│       └── forecast_future.png
│
└── README.md
```

---

##  Output

The project generates:

* Sales trend graph
* Monthly revenue graph
* Time series decomposition graph
* Actual vs predicted graph
* Error metrics graph
* 30-day future sales forecast graph

---

##  Model Used

This project uses the ARIMA model for time series forecasting.

ARIMA stands for:

* AutoRegressive (AR)
* Integrated (I)
* Moving Average (MA)

Model used:

```text id="o1mj4k"
ARIMA(5,1,0)
```

---

##  Evaluation Metrics

The model performance is evaluated using:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* MAPE (Mean Absolute Percentage Error)

---

##  Author

**Spandana H Y**
