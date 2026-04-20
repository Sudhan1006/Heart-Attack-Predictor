import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Setup
st.set_page_config(page_title="Heart Health AI", page_icon="❤️", layout="wide")
st.title("❤️ Heart Attack Risk Predictor")
st.markdown("### Analyze patient risk with high precision.")
st.info("💡 **New to this?** A detailed **User Guide** is provided at the bottom of this page to help you understand the input values.")

# 2. Load Model and Scaler
@st.cache_resource
def load_assets():
    model = pickle.load(open('heart_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("Error loading model files. Please ensure pkl files are in the repository.")

# 3. User Input Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Numerical Data")
    age = st.number_input("Age", min_value=1, max_value=110, value=50)
    gender = st.selectbox("Gender", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    chestpain = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
    restingBP = st.number_input("Resting Blood Pressure", value=130)
    serumcholestrol = st.number_input("Serum Cholestrol", value=240)
    fastingbloodsugar = st.selectbox("Fasting Blood Sugar > 120 (1=Yes, 0=No)", options=[0, 1])

with col2:
    st.subheader("Clinical Data")
    restingrelectro = st.selectbox("Resting ECG Results (0-2)", options=[0, 1, 2])
    maxheartrate = st.number_input("Maximum Heart Rate", value=150)
    exerciseangia = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", options=[0, 1])
    oldpeak = st.number_input("Oldpeak (ST depression)", value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST segment (0-3)", options=[0, 1, 2, 3])
    noofmajorvessels = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])

# 4. Prediction Logic
if st.button("Analyze Risk", use_container_width=True):
    features = [age, gender, chestpain, restingBP, serumcholestrol, fastingbloodsugar, 
                restingrelectro, maxheartrate, exerciseangia, oldpeak, slope, noofmajorvessels]
    
    cols = ['age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol', 'fastingbloodsugar', 
            'restingrelectro', 'maxheartrate', 'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels']
    
    input_df = pd.DataFrame([features], columns=cols)
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[:, 1][0]
    
    st.divider()
    if prob >= 0.45:
        st.error(f"🚨 **HIGH RISK Identified** (Probability: {prob:.2%})")
        st.warning("Recommendation: Please consult a cardiologist immediately for further clinical evaluation.")
    else:
        st.success(f"✅ **LOW RISK Identified** (Probability: {prob:.2%})")
        st.info("Recommendation: Maintain a healthy lifestyle and regular checkups.")

    # Visualization
    if hasattr(model, 'feature_importances_'):
        st.subheader("🔍 Why this result?")
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=cols).sort_values(ascending=True)
        fig, ax = plt.subplots()
        feat_imp.tail(7).plot(kind='barh', ax=ax, color='crimson')
        plt.title("Key Factors Influencing Your Result")
        st.pyplot(fig)

# --- 5. THE USER GUIDE (BOTTOM PART) ---
st.divider()
st.subheader("📘 User Guide: Understanding the Inputs")
st.markdown("""
Use the guide below to select the correct values based on clinical reports:

| Feature | Value | Meaning |
| :--- | :--- | :--- |
| **Chest Pain Type** | 0, 1, 2, 3 | 0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic |
| **Resting ECG** | 0, 1, 2 | 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy |
| **Exercise Angina** | 0, 1 | 0: No pain during exercise, 1: Chest pain induced by exercise |
| **Oldpeak** | Number | ST depression induced by exercise relative to rest (check ECG report) |
| **ST Slope** | 0, 1, 2, 3 | The slope of the peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping) |
| **Major Vessels** | 0-3 | Number of major vessels colored by flourosopy (0 means normal) |
""")
st.caption("Disclaimer: This tool is for educational purposes and based on machine learning patterns. It is not a substitute for professional medical advice.")
