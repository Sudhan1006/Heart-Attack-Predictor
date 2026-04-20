import streamlit as st
import pickleimport pickle
import numpy as np
import pandas as pd

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
    
    # ChestPainType: TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic
    chest_pain = st.selectbox("Chest Pain Type", options=["ATA", "NAP", "ASY", "TA"])
    
    # FastingBS: 1 if FastingBS > 120 mg/dl, 0 otherwise
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
    
    # RestingECG: Normal, ST, LVH
    resting_ecg = st.selectbox("Resting ECG Results", options=["Normal", "ST", "LVH"])
    
    # ExerciseAngina: Y: Yes, N: No
    ex_angina = st.selectbox("Exercise Induced Angina", options=["N", "Y"])
    
    # ST_Slope: Up, Flat, Down
    st_slope = st.selectbox("ST Slope Type", options=["Up", "Flat", "Down"])

# Dictionary based encoding (Model training ke logic ke hisaab se)
# Note: Agar tumne get_dummies use kiya tha, toh ye logic check karo:

# Sex_M: 1 for Male, 0 for Female
sex_m = 1 if sex == "M" else 0

# FastingBS
fbs_val = 1 if fasting_bs == "Yes" else 0

# ExerciseAngina_Y
angina_y = 1 if ex_angina == "Y" else 0

# ChestPainType (Assuming One-Hot Encoding order)
cp_ata = 1 if chest_pain == "ATA" else 0
cp_nap = 1 if chest_pain == "NAP" else 0
cp_asy = 1 if chest_pain == "ASY" else 0

# RestingECG
ecg_st = 1 if resting_ecg == "ST" else 0
ecg_lvh = 1 if resting_ecg == "LVH" else 0

# ST_Slope
slope_flat = 1 if st_slope == "Flat" else 0
slope_up = 1 if st_slope == "Up" else 0

# Sabko ek list mein dalo (ORDER MUST MATCH X_train columns!)
# Maan lo tumhara training order ye tha:
features_list = [
    age, resting_bp, cholesterol, fbs_val, max_hr, oldpeak,
    sex_m, cp_ata, cp_nap, cp_asy, ecg_st, ecg_lvh, angina_y, 
    slope_flat, slope_up
]

input_data = np.array([features_list])

if st.button("Analyze Risk"):
    # 1. Scaling 
    input_scaled = scaler.transform(input_data)
    
    # 2. find Probability
    prob = model.predict_proba(input_scaled)[:, 1]
    
    # 3. Custom Threshold (0.45)
    if prob >= 0.45:
        st.error(f"⚠️ High Risk! Probability: {prob[0]:.2f}")
        st.write("The patient shows symptoms consistent with heart disease.")
    else:
        st.success(f"✅ Low Risk. Probability: {prob[0]:.2f}")
        st.write("Everything looks normal, but always consult a doctor for certainty.")
     import matplotlib.pyplot as plt
import seaborn as sns

# Plotting Feature Importance
st.subheader("🔍 Model Decision Insights")
st.write("This chart shows which factors influenced the AI's decision the most.")

# extract importances from the model
importances = model.feature_importances_
feature_names = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
    'Sex_M', 'ChestPain_ATA', 'ChestPain_NAP', 'ChestPain_ASY', 
    'RestingECG_ST', 'RestingECG_LVH', 'ExerciseAngina_Y', 
    'ST_Slope_Flat', 'ST_Slope_Up'
]

# DataFrame 
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)

# Plotting logic
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax, palette='viridis')
plt.title("Top Contributing Factors")plt.title("Top Contributing Factors")

# Streamlit show
st.pyplot(fig)

st.write(f"Confidence Level: {prob[0]*100:.1f}%")
st.progress(float(prob[0])) # Blue bar will show probability
