from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'sales_forecast_secret_2024'

users = {}

arima_model  = pickle.load(open('arima_model.pkl',  'rb'))
daily_data   = pickle.load(open('daily_data.pkl',   'rb'))
arima_future = pickle.load(open('arima_future.pkl', 'rb'))
metrics      = pickle.load(open('arima_metrics.pkl','rb'))

os.makedirs('static/graphs', exist_ok=True)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET','POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        email    = request.form['email'].strip()
        password = request.form['password']
        if username in users:
            error = 'Username already exists.'
        elif not username or not email or not password:
            error = 'All fields are required.'
        else:
            users[username] = {'email': email, 'password': password}
            session['user'] = username
            return redirect(url_for('about'))
    return render_template('register.html', error=error)

@app.route('/login', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            session['user'] = username
            return redirect(url_for('about'))
        else:
            error = 'Invalid username or password.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/about')
def about():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('about.html', user=session['user'], metrics=metrics)

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
    result = None
    error  = None
    if request.method == 'POST':
        try:
            forecast_days = int(request.form.get('forecast_days', 30))
            forecast_days = max(7, min(forecast_days, 90))
            forecast      = np.array(arima_future.forecast(steps=forecast_days)).clip(0)
            last_date     = daily_data.index[-1]
            future_dates  = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
            forecast_df   = pd.DataFrame({
                'Date':             future_dates.strftime('%Y-%m-%d'),
                'Forecasted Sales': forecast.round(2)
            })
            plt.figure(figsize=(13,5))
            hist = min(90, len(daily_data))
            plt.plot(daily_data.index[-hist:], daily_data['Revenue'][-hist:],
                     label='Historical', color='#00e676', linewidth=1.5)
            plt.plot(future_dates, forecast, label=f'{forecast_days}-Day Forecast',
                     color='#00bcd4', linewidth=2, linestyle='--')
            plt.fill_between(future_dates, forecast*0.92, forecast*1.08,
                             alpha=0.2, color='#00bcd4', label='Confidence Band')
            plt.axvline(x=last_date, color='#ff6f00', linestyle=':', linewidth=1.5, label='Forecast Start')
            plt.title(f'{forecast_days}-Day Sales Forecast (ARIMA)', fontsize=13, fontweight='bold')
            plt.xlabel('Date'); plt.ylabel('Sales ($)')
            plt.legend(); plt.tight_layout()
            plt.savefig('static/graphs/dynamic_forecast.png', dpi=100, bbox_inches='tight')
            plt.close()
            result = {
                'days':  forecast_days,
                'total': f"{forecast.sum():,.2f}",
                'avg':   f"{forecast.mean():,.2f}",
                'max':   f"{forecast.max():,.2f}",
                'min':   f"{forecast.min():,.2f}",
                'table': forecast_df.head(15).to_dict('records'),
                'chart': '/static/graphs/dynamic_forecast.png'
            }
        except Exception as e:
            error = f'Forecasting failed: {str(e)}'
    return render_template('predict.html', user=session['user'],
                           result=result, error=error, metrics=metrics)

@app.route('/graphs')
def graphs():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('graphs.html', user=session['user'], metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)
