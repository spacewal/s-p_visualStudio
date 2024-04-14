import streamlit as st
from datetime import datetime
from data_handler import get_sp500_companies, get_sp500_tickers, fetch_data, prepare_data
from model import train_model
from visualizations import plot_predictions, plot_mplfinance

def analyze_stock(ticker):
    data = fetch_data(ticker, '2020-01-01', datetime.today().strftime('%Y-%m-%d'))
    data_scaled, scaler = prepare_data(data)

    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i, 0])
        y.append(data_scaled[i, 0])

    X_train, y_train = np.array(X[:int(len(X) * 0.8)]), np.array(y[:int(len(y) * 0.8)])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = train_model(X_train, y_train)

    latest_scaled = scaler.transform(data['Close'].values[-60:].reshape(-1, 1))
    latest_scaled = np.reshape(latest_scaled, (1, latest_scaled.shape[0], 1))
    predicted = model.predict(latest_scaled)
    fig = plot_mplfinance(data, predicted, scaler)
    st.pyplot(fig)
    plot_predictions(data, predicted, scaler)

def main():
    st.title("S&P 500 Stock Predictor")
    sp500_companies = get_sp500_companies()
    tickers = get_sp500_tickers()
    selected_ticker = st.selectbox("Choose a ticker to analyze", [''] + tickers)
    if selected_ticker:
        analyze_stock(selected_ticker)
    st.write("S&P 500 Dashboard")
    st.table(sp500_companies)

if __name__ == "__main__":
    main()
