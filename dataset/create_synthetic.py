"""
SYNTHETIC DATASET GENERATOR
Development/Testing Only — NOT real agricultural data.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

nitrogen = np.random.uniform(40, 140, n)
phosphorus = np.random.uniform(20, 80, n)
potassium = np.random.uniform(30, 100, n)
rainfall = np.random.uniform(800, 2500, n)
temperature = np.random.uniform(20, 35, n)
irrigation = np.random.uniform(200, 1000, n)
fertilizer = np.random.uniform(50, 300, n)
crop_duration = np.random.uniform(90, 150, n)
cultivated_area = np.random.uniform(0.5, 10, n)
ph = np.random.uniform(5.5, 7.5, n)
humidity = np.random.uniform(50, 90, n)
solar_radiation = np.random.uniform(10, 25, n)

# Yield formula (tonnes/hectare) with noise
# Scaled to produce realistic values 1-9 t/ha
yield_val = (
    0.020 * nitrogen
    + 0.015 * phosphorus
    + 0.010 * potassium
    + 0.0008 * rainfall
    + (-0.08 * (temperature - 27) ** 2)
    + 0.003 * irrigation
    + 0.006 * fertilizer
    + 0.010 * crop_duration
    + 0.04 * solar_radiation
    - 0.10 * cultivated_area
    + np.random.normal(0, 0.6, n)
)
# Normalise to realistic t/ha range
yield_val = yield_val - yield_val.min()
yield_val = yield_val / yield_val.max() * 7.5 + 1.5  # range ~1.5-9.0 t/ha
yield_val = np.clip(yield_val, 1.0, 10.0)

df = pd.DataFrame({
    'Nitrogen_kg_ha': np.round(nitrogen, 2),
    'Phosphorus_kg_ha': np.round(phosphorus, 2),
    'Potassium_kg_ha': np.round(potassium, 2),
    'Rainfall_mm': np.round(rainfall, 2),
    'Temperature_C': np.round(temperature, 2),
    'Irrigation_mm': np.round(irrigation, 2),
    'Fertilizer_kg_ha': np.round(fertilizer, 2),
    'Crop_Duration_days': np.round(crop_duration, 0).astype(int),
    'Cultivated_Area_ha': np.round(cultivated_area, 2),
    'Soil_pH': np.round(ph, 2),
    'Humidity_pct': np.round(humidity, 2),
    'Solar_Radiation_MJm2': np.round(solar_radiation, 2),
    'Rice_Yield_t_ha': np.round(yield_val, 3),
})

# Introduce some missing values (realistic)
for col in ['Nitrogen_kg_ha', 'Phosphorus_kg_ha', 'Rainfall_mm', 'Humidity_pct']:
    mask = np.random.choice([True, False], size=n, p=[0.03, 0.97])
    df.loc[mask, col] = np.nan

# Introduce a few duplicates
dupes = df.sample(5, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

df.to_csv('dataset/synthetic_rice_dataset.csv', index=False)
print(f"Synthetic dataset saved: {len(df)} rows, {len(df.columns)} columns")
print("Columns:", list(df.columns))
print("\nNOTE: This is a SYNTHETIC DEVELOPMENT/TEST dataset.")
print("Replace with your real CSV/XLSX for actual results.")
