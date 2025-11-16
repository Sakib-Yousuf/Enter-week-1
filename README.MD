# ⚡ Energy Usage Prediction Model

This project is a **Streamlit-based web application** that predicts **Energy Usage (kWh)** using a **pre-trained Random Forest Regression model**.  
The machine learning model is already trained and saved (`rf.pkl`), and the app provides a simple and interactive **frontend UI** to generate predictions based on input feature values.

---

## 📌 Overview

This application takes engineered time-series features such as lag values, rolling window statistics, and Fourier seasonal features as inputs, and generates a real-time prediction for energy usage.

The goal of this project is to make the model **accessible, interactive, and user-friendly** without needing to run any training code again.

---

## 🧠 Model Information

- **Type:** Regression Model  
- **Algorithm:** RandomForestRegressor  
- **Framework:** scikit-learn  
- **Saved File:** `rf.pkl` (loaded via joblib)  
- **Training Data:** `preprocessed_energy_data.csv`

The model was trained after applying:
- Lag features
- Rolling mean & standard deviation
- Outlier handling
- Fourier transformation for seasonality

---

## 📂 Project Structure

```
Project/
│
├── app.py                       # Streamlit frontend (UI)
├── models/
│   └── rf.pkl                   # Saved trained model
└── Dataset/
    └── preprocessed_energy_data.csv
```

---

## 🛠️ Requirements

Install all required dependencies using:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

---

## 🚀 Running the Application

Execute the following command in your terminal:

```bash
streamlit run app.py
```

Once executed, the web app will automatically open in your default browser.  
If not, manually visit the link displayed in the terminal (e.g. `http://localhost:8501`).

---

## 🔢 Input Features for Prediction

The app requires the following features to predict energy usage:

| Category | Features |
|----------|----------|
| Time-based | `Hour`, `DayOfWeek` |
| Lag Features | `lag_1`, `lag_2`, `lag__3`, `lag_6`, `lag_12`, `lag_24` |
| Rolling Window Mean | `roll_mean_3`, `roll_mean_6`, `roll_mean_12` |
| Rolling Window Std | `roll_std_3`, `roll_std_6`, `roll_std_12` |
| Seasonal Fourier | `sin_hour`, `cos_hour`, `sin_day`, `cos_day` |

These values can be entered manually using numeric input controls in the sidebar.

---

## 🧾 Application Features

✔ Loads saved ML model using joblib  
✔ Clean and responsive Streamlit interface  
✔ Real-time prediction  
✔ Numerical user input fields  
✔ No backend retraining or preprocessing required  

---




## ✨ Author

**Your Name Here**  
Machine Learning & Data Science Enthusiast  
