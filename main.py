import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Agri Price Prediction",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_lstm_model():
    return load_model("models/agri_price_prediction_model.h5")

model = load_lstm_model()

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/agri_prices.csv")
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], dayfirst=True)
    df = df.sort_values("Arrival_Date")
    return df

df = load_data()

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("🌱 Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Predict"])

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":
    st.title("🌾 Agricultural Price Prediction System")
    st.markdown("""
    This application predicts **future agricultural commodity prices**
    using a **deep learning LSTM model** trained on historical market data.

    ### Features
    - Real market data
    - LSTM-based time series forecasting
    - Interactive prediction interface
    """)

# -----------------------------
# ABOUT PAGE
# -----------------------------
elif page == "About":
    st.title("📘 About This Project")
    st.markdown("""
    **Technology Stack**
    - Python
    - TensorFlow (LSTM)
    - Streamlit
    - Pandas & NumPy

    **Use Case**
    - Helps farmers and traders estimate future prices
    - Supports better planning and decision-making

    **Model**
    - Long Short-Term Memory (LSTM)
    - Trained on historical modal prices
    """)

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "Predict":
    st.title("📈 Price Prediction")

    col1, col2 = st.columns([1, 2])

    with col1:
        commodity = st.selectbox(
            "Select Commodity",
            sorted(df["Commodity"].unique())
        )

        market = st.selectbox(
            "Select Market",
            sorted(df[df["Commodity"] == commodity]["Market"].unique())
        )

        days = st.slider("Days to Predict", 1, 30, 7)

        predict_btn = st.button("Predict Price")

    # -----------------------------
    # PREDICTION LOGIC (REAL)
    # -----------------------------
    if predict_btn:
        filtered = df[
            (df["Commodity"] == commodity) &
            (df["Market"] == market)
        ]

        if len(filtered) < 60:
            st.error("Not enough historical data for prediction.")
        else:
            prices = filtered["Modal Price"].values.reshape(-1, 1)

            # Scale data (same as training)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(prices)

            # Last 60 days
            last_60 = scaled_prices[-60:]
            input_seq = last_60.reshape(1, 60, 1)

            predictions = []
            future_dates = []

            last_date = filtered["Arrival_Date"].max()

            for i in range(days):
                next_scaled = model.predict(input_seq, verbose=0)[0][0]
                predictions.append(next_scaled)

                # Update sequence
                input_seq = np.append(
                    input_seq[:, 1:, :],
                    [[[next_scaled]]],
                    axis=1
                )

                future_dates.append(last_date + timedelta(days=i + 1))

            # Inverse scale
            predicted_prices = scaler.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            ).flatten()

            result_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Price (INR)": predicted_prices
            })

            with col2:
                st.subheader("📊 Prediction Results")
                st.line_chart(result_df.set_index("Date"))
                st.dataframe(result_df, use_container_width=True)
