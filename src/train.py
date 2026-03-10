# src/train.py

import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


def evaluate(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n📊 {model_name}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ── Modèles ────────────────────────────────────────────────────────
results = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
results["Linear Regression"] = evaluate("Linear Regression", y_test, y_pred_lr)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
results["Random Forest"] = evaluate("Random Forest", y_test, y_pred_rf)

# XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
results["XGBoost"] = evaluate("XGBoost", y_test, y_pred_xgb)

# ── Tuning ─────────────────────────────────────────────────────────
param_grid = {
    "n_estimators"    : [200, 300, 500],
    "max_depth"       : [4, 6, 8],
    "learning_rate"   : [0.01, 0.05, 0.1],
    "subsample"       : [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator  = xgb.XGBRegressor(random_state=42, n_jobs=-1),
    param_grid = param_grid,
    scoring    = "neg_root_mean_squared_error",
    cv         = 5,
    verbose    = 2,
    n_jobs     = -1
)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test_scaled)
results["XGBoost Tuned"] = evaluate("XGBoost Tuned", y_test, y_pred_best)

# ── Sérialisation ──────────────────────────────────────────────────
joblib.dump(best_model, "../models/xgb_model.pkl")
joblib.dump(scaler,     "../models/scaler.pkl")
print("\n✅ Modèle sauvegardé !")