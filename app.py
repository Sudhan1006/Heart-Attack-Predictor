import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Heart Health AI", page_icon="❤️")

# Title and Description
st.title("❤️ Heart Attack Risk Predictor")
st.write("Enter the patient's clinical details to analyze the risk.")

# Loading the saved files
model = pickle.load(open('heart_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Layout design
col1, col2 = st.columns(2)

with col1:
    st.subheader("Numerical Details")
    age = st.number_input("Age", min_value=1, max_value=110, value=40)
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=600, value=200)
    max_hr = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

with col2:
    st.subheader("Categorical Details")
    sex = st.selectbox("Sex", options=["M", "F"])
    
    # ChestPainType
    chest_pain = st.selectbox("Chest Pain Type", options=["ATA", "NAP", "ASY", "TA"])
    
    # FastingBS
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
    
    # RestingECG
    resting_ecg = st.selectbox("Resting ECG Results", options=["Normal", "ST", "LVH"])
    
    # ExerciseAngina
    ex_angina = st.selectbox("Exercise Induced Angina", options=["N", "Y"])
    
    # ST_Slope
    st_slope = st.selectbox("ST Slope Type", options=["Up", "Flat", "Down"])

# Encoding Logic
sex_m = 1 if sex == "M" else 0
fbs_val = 1 if fasting_bs == "Yes" else 0
angina_y = 1 if ex_angina == "Y" else 0
cp_ata = 1 if chest_pain == "ATA" else 0
cp_nap = 1 if chest_pain == "NAP" else 0
cp_asy = 1 if chest_pain == "ASY" else 0
ecg_st = 1 if resting_ecg == "ST" else 0
ecg_lvh = 1 if resting_ecg == "LVH" else 0
slope_flat = 1 if st_slope == "Flat" else 0
slope_up = 1 if st_slope == "Up" else 0

# Feature List (Must match training order)
features_list = [
    age, resting_bp, cholesterol, fbs_val, max_hr, oldpeak,
    sex_m, cp_ata, cp_nap, cp_asy, ecg_st, ecg_lvh, angina_y, 
    slope_flat, slope_up
]

input_data = np.array([features_list])

if st.button("Analyze Risk"):
    # 1. Scaling 
    input_scaled = scaler.transform(input_data)
    
    # 2. Probability
    prob = model.predict_proba(input_scaled)[:, 1]
    
    # 3. Output with 0.45 Threshold
    if prob[0] >= 0.45:
        st.error(f"⚠️ High Risk! Probability: {prob[0]:.2f}")
        st.write("The patient shows symptoms consistent with heart disease.")
    else:
        st.success(f"✅ Low Risk. Probability: {prob[0]:.2f}")
        st.write("Everything looks normal, but always consult a doctor for certainty.")

    # Visualization - Confidence Bar
    st.write(f"Confidence Level: {prob[0]*100:.1f}%")
    st.progress(float(prob[0]))

    # Feature Importance Plot
    st.markdown("---")
    st.subheader("🔍 Model Decision Insights")
    st.write("The chart below shows the most important factors for this prediction.")

    importances = model.feature_importances_
    feature_names = [
        'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'Sex_M', 'ChestPain_ATA', 'ChestPain_NAP', 'ChestPain_ASY', 
        'RestingECG_ST', 'RestingECG_LVH', 'ExerciseAngina_Y', 
        'ST_Slope_Flat', 'ST_Slope_Up'
    ]

    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)

    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax, palette='viridis')
    plt.title("Top Contributing Factors")
    st.pyplot(fig)
