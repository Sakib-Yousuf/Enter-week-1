import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# Load trained model
# ------------------------------
model = joblib.load("../models/model.pkl")

# ------------------------------
# Page Setup
# ------------------------------
st.set_page_config(page_title="Energy Usage Predictor", layout="wide")
st.title("⚡ Energy Usage Prediction")
st.write("Provide values below and get predicted Energy Usage (kWh).")

# ------------------------------
# Input Fields (Based on Model Features)
# ------------------------------

# IMPORTANT: Replace this with actual features from your model
required_features = [
    "Hour","DayOfWeek","lag_1","lag_2","lag_3","lag_6",
    "lag_12","lag_24","roll_mean_3","roll_mean_6","roll_mean_12",
    "roll_std_3","roll_std_6","roll_std_12","sin_hour","cos_hour",
    "sin_day","cos_day"
]

st.sidebar.header("📥 Enter Input Values")

input_values = {}
for feature in required_features:
    input_values[feature] = st.sidebar.number_input(f"{feature}", value=0.0, format="%.4f")

# Convert to DataFrame for prediction
user_df = pd.DataFrame([input_values])

# ------------------------------
# Prediction Button
# ------------------------------
if st.button("🔮 Predict"):
    prediction = model.predict(user_df)[0]
    st.success(f"Predicted Energy Usage: **{prediction:.3f} kWh**")

# ------------------------------
# Footer
# ------------------------------
st.write("---")
st.caption("Model Powered by RandomForestRegressor")


