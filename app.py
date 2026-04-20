import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Setup
st.set_page_config(page_title="Heart Attack Predictor", page_icon="❤️")
st.title("❤️ Heart Attack Risk Predictor")
st.write("Enter clinical details to analyze risk (Optimized for 100% Recall)")

# 2. Load Model and Scaler
@st.cache_resource # Taaki baar-baar load na ho
def load_assets():
    model = pickle.load(open('heart_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading files: Ensure 'heart_model.pkl' and 'scaler.pkl' are in the repository.")

# 3. User Input Layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=110, value=50)
    # gender: 1=Male, 0=Female (Matching your X_train)
    gender = st.selectbox("Gender", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    chestpain = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
    restingBP = st.number_input("Resting Blood Pressure", value=130)
    serumcholestrol = st.number_input("Serum Cholestrol", value=240)
    fastingbloodsugar = st.selectbox("Fasting Blood Sugar > 120 (1=Yes, 0=No)", options=[0, 1])

with col2:
    restingrelectro = st.selectbox("Resting Electrocardiographic results (0-2)", options=[0, 1, 2])
    maxheartrate = st.number_input("Maximum Heart Rate", value=150)
    exerciseangia = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", options=[0, 1])
    oldpeak = st.number_input("Oldpeak (ST depression)", value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST segment (0-3)", options=[0, 1, 2, 3])
    noofmajorvessels = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])

# 4. Prediction Logic
if st.button("Analyze Risk"):
    # Creating Feature List in EXACT order of your X_train columns
    features = [
        age, gender, chestpain, restingBP, serumcholestrol,
        fastingbloodsugar, restingrelectro, maxheartrate, 
        exerciseangia, oldpeak, slope, noofmajorvessels
    ]
    
    # Converting to DataFrame to keep feature names consistent
    cols = ['age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol',
            'fastingbloodsugar', 'restingrelectro', 'maxheartrate', 
            'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels']
    
    input_df = pd.DataFrame([features], columns=cols)
    
    # Scaling
    input_scaled = scaler.transform(input_df)
    
    # Predicting Probability for Custom Threshold (0.45)
    prob = model.predict_proba(input_scaled)[:, 1][0]
    
    # Result Display
    st.subheader("Result:")
    if prob >= 0.45:
        st.error(f"🚨 HIGH RISK (Probability: {prob:.2%})")
        st.write("The model predicts a high risk of heart attack. Please consult a specialist.")
    else:
        st.success(f"✅ LOW RISK (Probability: {prob:.2%})")
        st.write("The clinical indicators are within a safer range.")

    # 5. Visualization (Feature Importance)
    st.divider()
    st.subheader("🔍 Why this result?")
    try:
        # Checking if model has feature_importances_ (RandomForest/XGBoost)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=cols).sort_values(ascending=True)
            
            fig, ax = plt.subplots()
            feat_imp.tail(10).plot(kind='barh', ax=ax, color='crimson')
            plt.title("Top Factors Influencing Prediction")
            st.pyplot(fig)
    except:
        st.info("Feature importance visualization is available for Tree-based models.")
