import streamlit as st
from lstm_model import train_predict_plot

st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Prediction (LSTM)")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, INFY.NS):", "AAPL")
if st.button("Predict"):
    train_predict_plot(ticker)
