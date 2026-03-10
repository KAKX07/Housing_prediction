# src/preprocessing.py

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Step 1: Load the dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Step 2: Remove the MedHouseVal cap
df_clean = df[df["MedHouseVal"] < 5.0].copy()

# Step 3: Enrich the dataset
df_clean["rooms_per_household"]      = df_clean["AveRooms"]  / df_clean["AveOccup"]
df_clean["bedrooms_ratio"]           = df_clean["AveBedrms"] / df_clean["AveRooms"]
df_clean["population_per_household"] = df_clean["Population"] / df_clean["AveOccup"]

# Step 4: Split and scale
X = df_clean.drop(columns=["MedHouseVal"])
y = df_clean["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)