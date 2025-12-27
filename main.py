import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import os

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("🌾 Agricultural Commodity Price Prediction App")

# -----------------------------
# LOAD LSTM MODEL (SAFE)
# -----------------------------
@st.cache_resource
def load_lstm_model():
    model_path = "my_model.h5"

    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found. Please upload it to the repository.")
        st.stop()

    return load_model(model_path)

model = load_lstm_model()

# -----------------------------
# LOAD DATASET (SAFE)
# -----------------------------
@st.cache_data
def load_data():
    data_path = "Price_Agriculture_commodities_Week.csv"

    if not os.path.exists(data_path):
        st.error(f"❌ Dataset '{data_path}' not found. Please upload it to the repository.")
        st.stop()

    df = pd.read_csv(data_path)

    # Expected columns
    # Commodity | Market | Arrival_Date | Modal Price
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], dayfirst=True)
    df = df.sort_values("Arrival_Date")

    return df

df = load_data()

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
    max_value=100,
    value=7
)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.sidebar.button("🔮 Predict"):

    filtered_df = df[
        (df["Commodity"] == commodity) &
        (df["Market"] == market)
    ]

    if len(filtered_df) < 60:
        st.error("❌ Not enough historical data (minimum 60 records required).")
    else:
        prices = filtered_df["Modal Price"].values.reshape(-1, 1)

        # Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)

        # Last 60 time steps
        last_60 = scaled_prices[-60:]
        input_seq = last_60.reshape(1, 60, 1)

        predictions = []
        future_dates = []

        last_date = filtered_df["Arrival_Date"].max()

        for i in range(days):
            next_scaled = model.predict(input_seq, verbose=0)[0][0]
            predictions.append(next_scaled)

            # Update input sequence
            input_seq = np.append(
                input_seq[:, 1:, :],
                [[[next_scaled]]],
                axis=1
            )

            future_dates.append(last_date + timedelta(days=i + 7))  # weekly data

        # Inverse scaling
        predicted_prices = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()

        # -----------------------------
        # RESULT DISPLAY
        # -----------------------------
        st.subheader("📌 Prediction Result")

        avg_price = predicted_prices.mean()

        st.success(
            f"The predicted average price of **{commodity}** in **{market}** "
            f"for the next **{days} weeks** is approximately **₹{avg_price:.2f}**."
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
# INSTRUCTIONS
# -----------------------------
st.write("""
### ℹ️ Instructions
1. Select the commodity and market from the sidebar.
2. Choose the number of weeks to predict.
3. Click **Predict** to view future prices.
4. Predictions are generated using an **LSTM deep learning model** trained on weekly agricultural price data.
""")
