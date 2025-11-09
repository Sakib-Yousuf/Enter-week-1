# 🏢 Building Energy Data Analysis

This project analyzes building energy usage data using Python.  
It includes data cleaning, outlier detection, visualization, and encoding of categorical features.

## 📊 Features
- Loads and inspects dataset
- Detects missing values and outliers (using IQR)
- Visualizes data (histogram, boxplot, scatterplot, heatmap)
- Encodes categorical columns:
  - Label Encoding for `Building_Type`
  - Ordinal Encoding for `Occupancy_Level` (`Low < Medium < High`)

## 🧰 Libraries Used
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

## ⚙️ How to Run
1. Place your dataset at:  
   `../dataset/building_energy_data_extended.csv`
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
