# 🏢 Building Energy Consumption — Data Preparation & Analysis

Short project README for preprocessing and exploratory analysis of building energy usage.

## Overview
This repository contains a Jupyter notebook that performs data loading, inspection, visualization, outlier detection, and basic categorical encoding on an hourly building energy dataset.

## Files
- [notebook/preprocssing.ipynb](notebook/preprocssing.ipynb) — preprocessing notebook (load, EDA, plots, encoding)
- [dataset/building_energy_data_extended.csv](dataset/building_energy_data_extended.csv) — dataset (Timestamp, Building_ID, Energy_Usage (kWh), Temperature (°C), Humidity (%), Building_Type, Occupancy_Level)
- [requirements.txt](requirements.txt) — Python dependencies
- README.MD — this file

## What the notebook does
- Loads the CSV dataset and prints shape / info
- Summarizes numerical columns with df.describe().T
- Visualizes distributions and relationships:
  - Histograms for numerical columns
  - Boxplots per numerical column (IQR / outlier visualization)
  - Scatterplots (Temperature / Humidity vs Energy_Usage)
  - Correlation heatmap
- Detects outliers using IQR-based whisker function
- Encodes categorical features:
  - Label Encoding for `Building_Type`
  - Ordinal Encoding for `Occupancy_Level` in order `Low < Medium < High`

You can open the notebook here: [notebook/preprocssing.ipynb](notebook/preprocssing.ipynb)

## How to run
1. Create / activate your virtual environment (optional):
   - Windows (PowerShell): `.\mvenv\Scripts\Activate.ps1`
   - Windows (cmd): `.\mvenv\Scripts\activate.bat`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
