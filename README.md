# California Housing Price Predictor

An end-to-end Machine Learning project predicting California housing prices.
From data exploration to an interactive user interface built with Streamlit.

---

##  Goal

Predict the median housing price in California based on geographical and
demographic features, using the California Housing Dataset.

---

##  Tech Stack

- **Python 3.13**
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — data visualization
- **Scikit-learn** — preprocessing & modeling
- **XGBoost** — final model
- **SHAP** — model interpretability
- **Streamlit** — user interface
- **Joblib** — model serialization

---

##  Project Structure
```
Housing_prediction/
│
├── notebooks/
│   └── EDA.ipynb            # Exploration & modeling
│
├── src/
│   ├── preprocessing.py     # Cleaning + feature engineering
│   ├── train.py             # Training + tuning
│   └── evaluate.py          # Metrics + SHAP
│
├── models/                  # Serialized models (not versioned)
├── app.py                   # Streamlit interface
├── requirements.txt         # Dependencies
└── .gitignore
```

---

##  Results

| Model | RMSE | R² |
|-------|------|----|
| Linear Regression | ~0.60 | ~0.63 |
| Random Forest | ~0.46 | ~0.77 |
| XGBoost | ~0.43 | ~0.81 |
| **XGBoost Tuned** | **~0.41** | **~0.83** |

> The model explains **83% of the variance** in housing prices.
> Average prediction error : **~$41,000**.

---

##  Key Insights

- **Latitude & Longitude** are the most important features according to SHAP,
  despite low linear correlation — their signal is non-linear.
- **MedInc** (median income) is the strongest economic signal.
- **Feature engineering** (rooms_per_household, bedrooms_ratio) improves
  performance over raw features.

---

##  Getting Started

### Installation
```bash
git clone https://github.com/KAKX07/Housing_prediction.git
cd Housing_prediction
pip install -r requirements.txt
```

### Train the model
```bash
cd notebooks
jupyter notebook EDA.ipynb
```

### Run the app
```bash
streamlit run app.py
```

---

## 👤 Author

**Alexandre KOK**  
Data Science Student