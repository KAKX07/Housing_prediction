import streamlit as st
import joblib
import numpy as np

# ── Chargement du modèle et du scaler ──────────────────────────────
model  = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ── Configuration de la page ────────────────────────────────────────
st.set_page_config(
    page_title="🏠 Housing Price Predictor",
    layout="wide"
)

st.title("🏠 California Housing Price Predictor")
st.markdown("Estime le prix médian d'un logement en Californie.")

# ── Sidebar : inputs utilisateur ────────────────────────────────────
st.sidebar.header("📋 Caractéristiques du logement")

MedInc     = st.sidebar.slider("Revenu médian du quartier (x$10k)", 0.5, 15.0, 3.0)
HouseAge   = st.sidebar.slider("Age de la maison (années)", 1, 52, 20)
AveRooms   = st.sidebar.slider("Nombre moyen de pièces", 1.0, 20.0, 5.0)
AveBedrms  = st.sidebar.slider("Nombre moyen de chambres", 1.0, 5.0, 1.0)
Population = st.sidebar.number_input("Population du quartier", 100, 10000, 1000)
AveOccup   = st.sidebar.slider("Occupation moyenne", 1.0, 10.0, 3.0)
Latitude   = st.sidebar.slider("Latitude", 32.5, 42.0, 37.0)
Longitude  = st.sidebar.slider("Longitude", -124.5, -114.0, -120.0)

# ── Feature engineering (identique au notebook) ─────────────────────
rooms_per_household      = AveRooms / AveOccup
bedrooms_ratio           = AveBedrms / AveRooms
population_per_household = Population / AveOccup

# ── Assemblage du vecteur de features ───────────────────────────────
features = np.array([[
    MedInc, HouseAge, AveRooms, AveBedrms, Population,
    AveOccup, Latitude, Longitude,
    rooms_per_household, bedrooms_ratio, population_per_household
]])

# ── Prédiction ───────────────────────────────────────────────────────
features_scaled = scaler.transform(features)
prediction      = model.predict(features_scaled)[0]
prix_estime     = prediction * 100_000

# ── Affichage du résultat ────────────────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💰 Prix estimé", f"${prix_estime:,.0f}")
with col2:
    st.metric("📍 Localisation", f"{Latitude:.2f}°N, {Longitude:.2f}°W")
with col3:
    st.metric("🏡 Pièces / foyer", f"{rooms_per_household:.1f}")

st.markdown("---")
st.caption("Modèle : XGBoost tuné — R² ~0.83 sur le test set")