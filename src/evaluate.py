# src/evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Comparaison des modèles ────────────────────────────────────────
def plot_model_comparison(results):
    """Graphique de comparaison RMSE et R² de tous les modèles."""
    models = list(results.keys())
    rmses  = [results[m]["rmse"] for m in models]
    r2s    = [results[m]["r2"]   for m in models]
    colors = ["steelblue", "coral", "green", "darkgreen"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(models, rmses, color=colors)
    axes[0].set_title("RMSE (moins = mieux)")
    axes[0].set_ylabel("RMSE")

    axes[1].bar(models, r2s, color=colors)
    axes[1].set_title("R² (plus = mieux)")
    axes[1].set_ylabel("R²")

    plt.tight_layout()
    plt.show()


# ── SHAP ───────────────────────────────────────────────────────────
def plot_shap(model, X_train_scaled, X_test_scaled, X_test):
    """Génère les deux graphiques SHAP : bar et dot plot."""

    explainer   = shap.Explainer(model, X_train_scaled)
    shap_values = explainer(X_test_scaled)

    # Bar plot — importance globale
    print("📊 SHAP Bar Plot — Importance globale des features")
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    # Dot plot — direction de l'impact
    print("📊 SHAP Dot Plot — Impact détaillé")
    shap.summary_plot(shap_values, X_test)