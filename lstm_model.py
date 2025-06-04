import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st

def train_predict_plot(ticker):
    df = yf.download(ticker, start='2015-01-01', end='2024-12-31')
    
    if df.empty or 'Close' not in df:
        st.error("Invalid ticker symbol or no data found.")
        return

    # Preprocessing
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)

    if X.shape[0] == 0:
        st.error("Not enough data to make a prediction.")
        return

    # Define LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Prediction
    input_seq = data_scaled[-60:].reshape(1, 60, 1)
    prediction = model.predict(input_seq)
    predicted_price = scaler.inverse_transform(prediction)

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(df['Close'], label="Historical Price")
    plt.axhline(y=predicted_price[0][0], color='r', linestyle='--',
                label=f"Predicted: ₹{predicted_price[0][0]:.2f}")
    plt.title(f"{ticker} Stock Price & Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True)

    # Show on Streamlit
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close()
