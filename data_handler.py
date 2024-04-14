import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def get_sp500_companies():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(url, header=0)[0]
    return df[['Symbol', 'GICS Sector']]

def get_sp500_tickers():
    df = get_sp500_companies()
    return df['Symbol'].tolist()

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.dropna(inplace=True)
    return data

def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return data_scaled, scaler
