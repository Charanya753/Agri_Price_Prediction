import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import os

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("🌾 Agricultural Price Prediction App")

# -----------------------------
# LOAD MODEL (SAFE)
# -----------------------------
try:
    model = load_model("agri_price_prediction_model.h5")
except FileNotFoundError:
    st.error("Model file 'agri_price_prediction_model.h5' not found. Please upload it to the repository.")
    st.stop()

# -----------------------------
# LOAD DATA (SAFE)
# -----------------------------
try:
    df = pd.read_csv("agri_prices.csv")
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], dayfirst=True)
    df = df.sort_values("Arrival_Date")
except FileNotFoundError:
    st.error("Dataset file 'agri_prices.csv' not found. Please upload it to the repository.")
    st.stop()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("📥 Enter Market Details")

commodity = st.sidebar.selectbox(
    "Commodity",
    sorted(df["Commodity"].unique())
)

market = st.sidebar.selectbox(
    "Market",
    sorted(df[df["Commodity"] == commodity]["Market"].unique())
)

days = st.sidebar.slider(
    "Days to Predict",
    min_value=1,
    max_value=30,
    value=7
)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.sidebar.button("Predict"):

    filtered = df[
        (df["Commodity"] == commodity) &
        (df["Market"] == market)
    ]

    if len(filtered) < 60:
        st.error("Not enough historical data for prediction (minimum 60 days required).")
    else:
        prices = filtered["Modal Price"].values.reshape(-1, 1)

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)

        # Last 60 days as input
        last_60 = scaled_prices[-60:]
        input_seq = last_60.reshape(1, 60, 1)

        predictions = []
        future_dates = []

        last_date = filtered["Arrival_Date"].max()

        for i in range(days):
            next_scaled = model.predict(input_seq, verbose=0)[0][0]
            predictions.append(next_scaled)

            input_seq = np.append(
                input_seq[:, 1:, :],
                [[[next_scaled]]],
                axis=1
            )

            future_dates.append(last_date + timedelta(days=i + 1))

        # Inverse scaling
        predicted_prices = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()

        # -----------------------------
        # DISPLAY RESULT
        # -----------------------------
        st.subheader("📌 Prediction Result")

        avg_price = predicted_prices.mean()

        st.write(
            f"The predicted average price of **{commodity}** in **{market}** "
            f"for the next **{days} days** is **₹{avg_price:.2f}**."
        )

        # -----------------------------
        # GRAPH
        # -----------------------------
        result_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price (INR)": predicted_prices
        })

        st.line_chart(result_df.set_index("Date"))

        # -----------------------------
        # TABLE
        # -----------------------------
        st.dataframe(result_df, use_container_width=True)

# -----------------------------
# INSTRUCTIONS (LIKE SLEEP APP)
# -----------------------------
st.write("""
### Instructions
1. Use the sidebar to select the commodity and market.
2. Choose the number of days for prediction.
3. Click **Predict** to view forecasted prices.
4. Predictions are generated using an LSTM deep learning model trained on historical data.
""")
