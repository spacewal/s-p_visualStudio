import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

def plot_predictions(data, predicted, scaler):
    predicted_prices = scaler.inverse_transform(predicted)
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'].tail(60).index, data['Close'].tail(60).values, color='blue', label='Actual Prices')
    predicted_dates = pd.date_range(start=data.tail(1).index[0], periods=len(predicted), freq='B')
    plt.plot(predicted_dates, predicted_prices, color='red', label='Predicted Prices')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_mplfinance(data, predicted, scaler):
    predicted_prices = scaler.inverse_transform(predicted)
    mpf_figure = mpf.plot(data[-60:], type='candle', style='charles', volume=True, returnfig=True)
    ax1 = mpf_figure[0].axes[0]
    predicted_dates = pd.date_range(start=data.tail(1).index[0] + pd.Timedelta(days=1), periods=len(predicted), freq='B')
    ax1.plot(predicted_dates, predicted_prices, color='red', label='Predicted Prices')
    ax1.legend()
    return mpf_figure[0]
