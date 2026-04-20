# Heart-Attack-Predictor
# ❤️ Heart Attack Risk Predictor (100% Recall Optimized)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://heart-attack-predictor-yxr7pzzipkdxuyw96arskp.streamlit.app/)

## 📌 Project Overview
This is a Machine Learning-based web application that predicts the risk of heart attack based on clinical parameters. The project was developed with a **Medical-First approach**, where the primary goal was to minimize False Negatives (missing a high-risk patient).

### 🚀 Live Demo
Check out the live app here: [Heart Attack Predictor](https://heart-attack-predictor-yxr7pzzipkdxuyw96arskp.streamlit.app/)

---

## 🎯 Key Highlights
- **100% Recall Strategy:** Tuned the model threshold to **0.45** to ensure that every high-risk patient is identified, making it safer for medical screening.
- **Explainable AI:** Integrated feature importance visualization to show *why* the model predicted a specific result.
- **User-Friendly UI:** Includes a detailed **User Guide** within the app to help non-technical users understand clinical input values.

---

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Model:** Random Forest Classifier (Hyperparameter Tuned)
- **Deployment:** Streamlit Cloud

---

## 📊 Dataset Features
The model uses 12 key clinical features:
1. `age` - Age of the patient
2. `gender` - Male/Female
3. `chestpain` - Type of chest pain (0-3)
4. `restingBP` - Blood pressure at rest
5. `serumcholestrol` - Cholesterol levels
6. `fastingbloodsugar` - Sugar levels (>120 mg/dl)
7. `restingrelectro` - ECG results
8. `maxheartrate` - Peak heart rate
9. `exerciseangia` - Pain during exercise
10. `oldpeak` - ST depression
11. `slope` - Slope of ST segment
12. `noofmajorvessels` - Number of major vessels (0-3)

---

## 📈 Performance Metrics
| Metric | Score |
| :--- | :--- |
| **Accuracy** | ~92% |
| **Recall (Sensitivity)** | **100%** |
| **Optimized Threshold** | 0.45 |

---

## 📂 Project Structure
```text
├── app.py              # Streamlit Web App code
├── heart_model.pkl     # Trained Random Forest Model
├── scaler.pkl          # Saved StandardScaler object
├── requirements.txt    # Library dependencies
└── README.md           # Project Documentation
