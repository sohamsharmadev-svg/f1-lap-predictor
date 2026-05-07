import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
le_driver = pickle.load(open('le_driver.pkl', 'rb'))
le_country = pickle.load(open('le_country.pkl', 'rb'))
df = pd.read_csv('f1_cleaned.csv')

st.set_page_config(page_title="F1 Lap Time Predictor", page_icon="🏎️", layout="wide")

st.title("🏎️ F1 Lap Time Predictor")
st.markdown("Predict Formula 1 lap times using Machine Learning (XGBoost)")

# ---- Sidebar inputs ----
st.sidebar.header("🔧 Race Parameters")

year = st.sidebar.slider("Season Year", 2011, 2020, 2018)
round_num = st.sidebar.slider("Race Round", 1, 21, 5)
lap = st.sidebar.slider("Lap Number", 1, 70, 20)
position = st.sidebar.slider("Current Position", 1, 20, 5)

driver = st.sidebar.selectbox("Driver", sorted(le_driver.classes_))
country = st.sidebar.selectbox("Circuit Country", sorted(le_country.classes_))

# Get circuit coordinates from data
circuit_info = df[df['country'] == country][['lat', 'lng', 'alt']].iloc[0]
lat, lng, alt = circuit_info['lat'], circuit_info['lng'], circuit_info['alt']

# ---- Prediction ----
driver_enc = le_driver.transform([driver])[0]
country_enc = le_country.transform([country])[0]

input_data = pd.DataFrame([[year, round_num, lap, position, lat, lng, alt, driver_enc, country_enc]],
    columns=['year', 'round', 'lap', 'position', 'lat', 'lng', 'alt', 'driver_encoded', 'country_encoded'])

prediction = model.predict(input_data)[0]

# ---- Display Results ----
col1, col2, col3 = st.columns(3)
col1.metric("⏱️ Predicted Lap Time", f"{prediction:.2f} sec")
col2.metric("🏁 Driver", driver.upper())
col3.metric("🌍 Country", country)

st.divider()

# ---- SHAP Explainability ----
st.subheader("🧠 Why this prediction? (SHAP Explanation)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_data)

fig, ax = plt.subplots(figsize=(10, 4))
shap.bar_plot(shap_values[0],
    feature_names=['year','round','lap','position','lat','lng','alt','driver','country'],
    show=False)
plt.tight_layout()
st.pyplot(fig)

st.divider()

# ---- Historical comparison ----
st.subheader("📊 Historical Lap Times for This Country")
country_data = df[df['country'] == country]['lap_time_sec']
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.hist(country_data, bins=40, color='red', alpha=0.7)
ax2.axvline(prediction, color='yellow', linewidth=2, label=f'Your prediction: {prediction:.2f}s')
ax2.set_xlabel("Lap Time (seconds)")
ax2.set_ylabel("Frequency")
ax2.legend()
st.pyplot(fig2)