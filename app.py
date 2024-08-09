import streamlit as st

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import date, timedelta,datetime
today = date.today()


st.title("BITCOIN PRICE PREDICTION")
number = st.number_input("Select a number of previous days to consider for prediction:", min_value=10, max_value=1000, value=50)


d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=number)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('BTC-USD', 
start=start_date, 
end=end_date, 
progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)


import plotly.graph_objects as go
figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"], 
                                        high=data["High"],
                                        low=data["Low"], 
                                        close=data["Close"])])
figure.update_layout(title = "Bitcoin Price Analysis", 
                     xaxis_rangeslider_visible=False)

st.plotly_chart(figure)

from statsmodels.tsa.arima.model import ARIMA

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data = data.asfreq('D') 

model = ARIMA(data['Close'], order=(5, 1, 0))
model_fit = model.fit()

st.write("The prediction by the model for next 30 days is Following :")
forecast = model_fit.forecast(steps=30)
fig = go.Figure(data=go.Scatter(x=forecast.index, y=forecast, mode='lines+markers'))
fig.update_layout(title='Line Plot', xaxis_title='X-axis Label', yaxis_title='Y-axis Label')

st.plotly_chart(fig)