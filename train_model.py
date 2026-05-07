import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import pickle

# Load cleaned data
df = pd.read_csv('f1_cleaned.csv')

# Encode text columns to numbers (ML needs numbers)
le_driver = LabelEncoder()
le_country = LabelEncoder()

df['driver_encoded'] = le_driver.fit_transform(df['driverRef'])
df['country_encoded'] = le_country.fit_transform(df['country'])

# Features (inputs) and Target (what we predict)
features = ['year', 'round', 'lap', 'position', 'lat', 'lng', 'alt', 'driver_encoded', 'country_encoded']
target = 'lap_time_sec'

X = df[features]
y = df[target]

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} laps, testing on {len(X_test)} laps...")

# Train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n✅ Model trained!")
print(f"📊 Mean Absolute Error: {mae:.2f} seconds")
print(f"📊 R² Score: {r2:.4f}  (1.0 = perfect)")

# Save model and encoders for the dashboard
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(le_driver, open('le_driver.pkl', 'wb'))
pickle.dump(le_country, open('le_country.pkl', 'wb'))

print("\n💾 Model saved to model.pkl!")