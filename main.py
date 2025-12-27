import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Agricultural Commodity Price Prediction",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Agricultural Commodity Price Prediction App")

# -----------------------------
# LOAD LSTM MODEL
# -----------------------------
@st.cache_resource
def load_lstm_model():
    model_path = "my_model.h5"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found.")
        st.stop()
    return load_model(model_path)

model = load_lstm_model()

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():
    data_path = "Price_Agriculture_commodities_Week.csv"
    if not os.path.exists(data_path):
        st.error(f"❌ Dataset '{data_path}' not found.")
        st.stop()

    df = pd.read_csv(data_path)
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

weeks = st.sidebar.slider(
    "Weeks to Predict",
    min_value=1,
    max_value=100,
    value=7
)

# -----------------------------
# PREDICTION LOGIC
# -----------------------------
if st.sidebar.button("🔮 Predict"):

    # Market-level filtering
    filtered_df = df[
        (df["Commodity"] == commodity) &
        (df["Market"] == market)
    ]

    # Silent fallback to commodity-level data
    if len(filtered_df) < 10:
        filtered_df = df[df["Commodity"] == commodity]

    total_records = len(filtered_df)
    time_steps = min(60, total_records)

    if time_steps < 10:
        st.error(
            f"❌ Not enough historical data for prediction. "
            f"Only {total_records} records available."
        )
        st.stop()

    # Prepare price data
    prices = filtered_df["Modal Price"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    last_seq = scaled_prices[-time_steps:]
    input_seq = last_seq.reshape(1, time_steps, 1)

    predictions = []
    future_dates = []

    last_date = filtered_df["Arrival_Date"].max()

    # Predict future prices
    for i in range(weeks):
        next_scaled = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(next_scaled)

        input_seq = np.append(
            input_seq[:, 1:, :],
            [[[next_scaled]]],
            axis=1
        )

        # Weekly data → add 7 days
        future_dates.append(last_date + timedelta(days=(i + 1) * 7))

    # Inverse scaling
    predicted_prices = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    # -----------------------------
    # RESULTS
    # -----------------------------
    st.subheader("📌 Prediction Result")

    st.success(
        f"The predicted average price of **{commodity}** "
        f"for the next **{weeks} weeks** is approximately "
        f"**₹{predicted_prices.mean():.2f}**."
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
