import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and features
with open("models/temperature_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/temperature_features.pkl", "rb") as f:
    feature_list = pickle.load(f)

st.title("Weather Predictor")
st.markdown("Predict tomorrow's temperature based on today's weather in London.")

st.divider()

# User inputs
col1, col2 = st.columns(2)

with col1:
    max_temp = st.number_input("Max Temperature (°C)", value=10.0, key="max_temp")
    min_temp = st.number_input("Min Temperature (°C)", value=5.0, key="min_temp")
    cloud_cover = st.number_input("Cloud Cover (%)", value=50, min_value=0, max_value=100, key="cloud_cover")

with col2:
    date = st.date_input("Date")
    year = date.year
    pressure = st.number_input("Pressure (mbar)", value=1013, key="pressure")

# Past 5 days temps
st.markdown("#### Last 5 Days Mean Temperatures (°C)")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    last1 = st.number_input("Day 1", value=8.0, key="last1", label_visibility="collapsed")
with col2:
    last2 = st.number_input("Day 2", value=7.0, key="last2", label_visibility="collapsed")
with col3:
    last3 = st.number_input("Day 3", value=8.0, key="last3", label_visibility="collapsed")
with col4:
    last4 = st.number_input("Day 4", value=9.0, key="last4", label_visibility="collapsed")
with col5:
    last5 = st.number_input("Day 5", value=8.0, key="last5", label_visibility="collapsed")

# Predict button
if st.button("Predict Tomorrow's Temperature", use_container_width=True):
    # Calculate derived features
    mean_temp = (max_temp + min_temp) / 2
    temp_last5 = np.mean([last1, last2, last3, last4, last5])
    pressure_pa = pressure * 100
    cloud_oktas = round(cloud_cover / 100 * 8)

    # Create input dataframe
    input_data = pd.DataFrame([{
        "max_temp": max_temp,
        "mean_temp": mean_temp,
        "cloud_cover": cloud_oktas,
        "min_temp": min_temp,
        "temp_last5": temp_last5,
        "year": year,
        "pressure": pressure_pa
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.divider()
    st.metric("Predicted Temperature Tomorrow", f"{prediction:.2f} °C")
