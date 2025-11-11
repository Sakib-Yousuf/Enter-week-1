# Building Energy Consumption — Forecasting Prototype

A small project that trains a simple regression model to predict building energy usage (kWh) from environmental and building features. The repository contains datasets and Jupyter notebooks for preprocessing and modelling.

## Key points
- Input features used by the current model: `Temperature (°C)`, `Humidity (%)`, `Building_Type`, `Occupancy_Level`.
- Target: `Energy_Usage (kWh)`.
- Current model (in `notebook/model.ipynb`): RandomForestRegressor (n_estimators=300, random_state=42).

## Repository structure
- `dataset/` — CSV files used by notebooks (e.g., `processed_data.csv`, `Raw_data.csv`).
- `notebook/preprocssing.ipynb` — preprocessing and exploratory data analysis.
- `notebook/model.ipynb` — model training, prediction, and basic evaluation (MSE, MAE, R²).
- `requirements.txt` — Python dependencies to install.

## How the model notebook is organized
1. Load `dataset/processed_data.csv`.
2. Select features and target:
   - X = `['Temperature (°C)', 'Humidity (%)', 'Building_Type', 'Occupancy_Level']`
   - y = `Energy_Usage (kWh)`
3. Train/Test split (test_size=0.2, random_state=42).
4. Train a RandomForestRegressor (n_estimators=300).
5. Predict on test set and print evaluation metrics: MSE, MAE, R².

## Quick start (Windows PowerShell)
1. Create & activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Open the notebooks in Jupyter Lab / Notebook and run the cells:

```powershell
jupyter lab
```

Then open `notebook/preprocssing.ipynb` and `notebook/model.ipynb`.

## Reproducing the training run (notes)
- The model notebook trains the RandomForest on the preprocessed CSV. There is no separate training script; use the notebook to run training and evaluation.
- If your `Building_Type` or `Occupancy_Level` are categorical strings, ensure they are encoded before training (the notebooks handle simple encodings). If you run training from a script, ensure consistent encoding for train and inference.

## Results & evaluation
- The notebook prints MSE, MAE, and R² for the test set (look at the last cells in `notebook/model.ipynb`).

## Next steps & improvements
- Add a dedicated training script (e.g., `train.py`) to allow CLI runs and reproducible checkpoints.
- Save the trained model with `joblib` or `pickle` and provide an inference example script.
- Improve preprocessing: one-hot or target encoding for `Building_Type`, consistent ordinal mapping for `Occupancy_Level`.
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) and cross-validation.
- Add unit tests for preprocessing functions and a small CI workflow.

## Dependencies
Install from `requirements.txt`. It likely contains packages such as `pandas`, `scikit-learn`, `matplotlib` / `seaborn`, and `jupyter`.

