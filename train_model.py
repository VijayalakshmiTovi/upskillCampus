
import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, "data", "crop_production_starter_dataset.csv"
)
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully:", df.shape)

# Clean dataset
df.columns = df.columns.str.strip()

required_columns = [
    "State", "District", "Crop_Year",
    "Season", "Crop", "Area", "Production"
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

df = df[required_columns].copy()
df = df.dropna()

df["Crop_Year"] = pd.to_numeric(df["Crop_Year"], errors="coerce")
df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
df["Production"] = pd.to_numeric(df["Production"], errors="coerce")

df = df.dropna()
df = df[
    (df["Area"] > 0) &
    (df["Production"] >= 0)
]

if len(df) < 5:
    raise ValueError("Not enough valid rows to train the model.")

# Input features and prediction target
features = [
    "State", "District", "Crop_Year",
    "Season", "Crop", "Area"
]

X = df[features]
y = df["Production"]

categorical = ["State", "District", "Season", "Crop"]
numeric = ["Crop_Year", "Area"]

# Prepare categorical and numerical columns
preprocessor = ColumnTransformer([
    ("categories", OneHotEncoder(handle_unknown="ignore"),
     categorical),
    ("numbers", "passthrough", numeric)
])

# Create model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=2
    ))
])

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
print("Training the Random Forest model...")
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n--- MODEL EVALUATION ---")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 3))

# Save trained model
model_path = os.path.join(MODEL_DIR, "crop_model.joblib")
joblib.dump(model, model_path)

# Save evaluation metrics
metrics = {
    "model": "Random Forest Regressor",
    "rows_used": int(len(df)),
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
    "dataset_note": (
        "Synthetic starter data; metrics are for pipeline testing only."
    )
}

metrics_path = os.path.join(MODEL_DIR, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nModel training completed!")
print("Saved model:", model_path)
print("Saved metrics:", metrics_path)
