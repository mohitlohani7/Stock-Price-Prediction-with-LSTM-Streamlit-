import streamlit as st
from lstm_model import train_predict_plot

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="centered"
)

# --- Stylish Title ---
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
    }
    .sub-text {
        text-align: center;
        font-size: 1.2em;
        color: #444;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Stock Price Predictor using LSTM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Enter a stock ticker (e.g., AAPL, TSLA, INFY) to forecast the next price</div>', unsafe_allow_html=True)

# --- User Input ---
with st.form("prediction_form"):
    ticker = st.text_input("🔍 Stock Ticker Symbol", value="AAPL", max_chars=10)
    submitted = st.form_submit_button("Predict Stock Price 📈")

# --- Prediction & Plot ---
if submitted:
    with st.spinner("🔄 Fetching data and predicting... Please wait."):
        train_predict_plot(ticker.upper())

# --- Footer ---
st.markdown("""
    <hr>
    <center>
        <small>Made with ❤️ using Streamlit & LSTM | <a href="https://github.com/mohitlohani7/stock-price-prediction-with-lstm-streamlit-" target="_blank">GitHub Repo</a></small>
    </center>
""", unsafe_allow_html=True)
