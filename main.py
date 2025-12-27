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
    page_title="Agri Price Prediction App",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_lstm_model():
    return load_model("my_model.h5")

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

# =============================
# SIDEBAR – INPUTS
# =============================
st.sidebar.header("🌱 Enter Market Details")

commodity = st.sidebar.selectbox(
    "Commodity",
    sorted(df["Commodity"].unique())
)

market = st.sidebar.selectbox(
    "Market",
    sorted(df[df["Commodity"] == commodity]["Market"].unique())
)

days = st.sidebar.slider(
    "Prediction Duration (days)",
    1, 30, 7
)

predict_btn = st.sidebar.button("🔮 Predict")

# =============================
# MAIN PAGE
# =============================
st.title("🌾 Agricultural Price Prediction App")

st.markdown(
    "This application predicts **future agricultural commodity prices** "
    "using a trained **LSTM deep learning model**."
)

# =============================
# PREDICTION RESULT
# =============================
if predict_btn:

    filtered = df[
        (df["Commodity"] == commodity) &
        (df["Market"] == market)
    ]

    if len(filtered) < 60:
        st.error("❌ Not enough historical data for prediction.")
    else:
        prices = filtered["Modal Price"].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)

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

        predicted_prices = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()

        # -----------------------------
        # RESULT TEXT (Like Sleep App)
        # -----------------------------
        st.subheader("📌 Prediction Result")

        avg_price = predicted_prices.mean()

        st.success(
            f"The predicted average price of **{commodity}** in **{market}** "
            f"for the next **{days} days** is approximately **₹{avg_price:.2f}**."
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
        # INSTRUCTIONS (Like Sleep App)
        # -----------------------------
        st.subheader("ℹ️ Instructions")
        st.markdown("""
        1. Use the sidebar to select commodity and market.
        2. Choose prediction duration.
        3. Click **Predict** to view forecasted prices.
        4. Predictions are based on historical trends using LSTM.
        """)

else:
    st.info("👈 Use the sidebar to enter details and click **Predict**")
