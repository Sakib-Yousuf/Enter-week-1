import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ===============================
# 1. LOAD MODEL & FEATURES
# ===============================
model = joblib.load("../models/model.pkl")
feature_cols = joblib.load("../models/feature_cols.pkl")  # saved column order

st.set_page_config(page_title="Energy Usage Predictor", layout="wide")
st.title("🏢 Building Energy Usage Prediction Dashboard")

# ===============================
# 2. UPLOAD DATA OR INPUT MANUALLY
# ===============================
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV file for prediction", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("Manual Input (optional)")

# Placeholder for manual input
manual_input = {}
for f in feature_cols:
    manual_input[f] = st.sidebar.number_input(f, value=0.0)

# ===============================
# 3. PREDICTION LOGIC
# ===============================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Ensure feature columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in uploaded file: {missing_cols}")
    else:
        X = df[feature_cols]
        y_pred = model.predict(X)
        df["Predicted_Energy_Usage"] = y_pred

        st.subheader("Predictions")
        st.dataframe(df.head(20))

        # Plot actual vs predicted if Energy_Usage exists
        if "Energy_Usage (kWh)" in df.columns:
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(df.index, df["Energy_Usage (kWh)"], label="Actual")
            ax.plot(df.index, df["Predicted_Energy_Usage"], label="Predicted")
            ax.set_xlabel("Index")
            ax.set_ylabel("Energy Usage (kWh)")
            ax.legend()
            st.pyplot(fig)

else:
    st.warning("Upload a CSV to get predictions, or enter manual input below.")

    # Convert manual input dict to DataFrame
    X_manual = pd.DataFrame([manual_input])
    y_manual_pred = model.predict(X_manual)[0]

    st.subheader("Predicted Energy Usage")
    st.metric("Energy_Usage (kWh)", f"{y_manual_pred:.2f}")

# ===============================
# 4. FOOTER
# ===============================
st.markdown("---")
st.markdown("Powered by Random Forest Regression | EDUNET FOUNDATION Internship")
