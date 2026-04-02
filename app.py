import streamlit as st
import pandas as pd
import pickle

# =========================
# Load Model and Scaler
# =========================
with open("heart-disease-model.pkl", "rb") as file:
    model, scaler, expected_encoded = pickle.load(file)

# Feature lists
categorical_cols = ['Gender', 'ChestPainType', 'FastingBS', 'RestingECG',
                    'ExerciseAngina', 'ST_Slope', 'MajorVessels', 'Thalassemia']

numerical_cols = ['Age', 'Cholesterol', 'RestingBP', 'MaxHR', 'ST_Depression']

# Gives the original list of columns on which the model is trained
expected_encoded = model.feature_names_in_

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Heart Disease Prediction")
st.title("❤️ Heart Disease Prediction")
st.write("""
This app predicts the likelihood of heart disease based on patient medical data.
""")
st.warning("""
⚠️ **Important:** This prediction is for educational purposes only and **does not replace professional medical advice**.  
Please consult a qualified doctor for a proper diagnosis and treatment.
""")
st.markdown("---")

col1, col2 = st.columns(2)

# Mapping dictionaries
gender_map = {"Female": 0, "Male": 1}
chest_pain_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}
fasting_bs_map = {"No": 0, "Yes": 1}
resting_ecg_map = {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
exercise_angina_map = {"No": 0, "Yes": 1}
st_slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thalassemia_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
major_vessels_map = {"0": 0, "1": 1, "2": 2, "3": 3}

with col1:
    Age = st.number_input("Age (years)", 10, 100, 45)
    Gender_display = st.selectbox("Gender", list(gender_map.keys()))
    ChestPainType_display = st.selectbox("Chest Pain Type", list(chest_pain_map.keys()))
    RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    Cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    FastingBS_display = st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(fasting_bs_map.keys()))

with col2:
    RestingECG_display = st.selectbox("Resting ECG Results", list(resting_ecg_map.keys()))
    MaxHR = st.number_input("Maximum Heart Rate (bpm)", 60, 220, 150)
    ExerciseAngina_display = st.selectbox("Exercise-Induced Angina", list(exercise_angina_map.keys()))
    ST_Depression = st.number_input("ST Depression (mm)", 0.0, 6.0, 1.0, step=0.1)
    ST_Slope_display = st.selectbox("ST Slope", list(st_slope_map.keys()))
    MajorVessels_display = st.selectbox("Major Vessels", list(major_vessels_map.keys()),help="Number of major vessels colored by fluoroscopy")
    Thalassemia_display = st.selectbox("Thalassemia", list(thalassemia_map.keys()))

# Convert all user-friendly selections to numeric values for the model
Gender = gender_map[Gender_display]
ChestPainType = chest_pain_map[ChestPainType_display]
FastingBS = fasting_bs_map[FastingBS_display]
RestingECG = resting_ecg_map[RestingECG_display]
ExerciseAngina = exercise_angina_map[ExerciseAngina_display]
ST_Slope = st_slope_map[ST_Slope_display]
MajorVessels = major_vessels_map[MajorVessels_display]
Thalassemia = thalassemia_map[Thalassemia_display]

# =========================
# Create Input DataFrame
# =========================
input_dict = {
    "Age": Age,
    "Gender": Gender,
    "ChestPainType": ChestPainType,
    "RestingBP": RestingBP,
    "Cholesterol": Cholesterol,
    "FastingBS": FastingBS,
    "RestingECG": RestingECG,
    "MaxHR": MaxHR,
    "ExerciseAngina": ExerciseAngina,
    "ST_Depression": ST_Depression,
    "ST_Slope": ST_Slope,
    "MajorVessels": MajorVessels,
    "Thalassemia": Thalassemia
}

input_df = pd.DataFrame([input_dict])

# One-hot encode categorical columns
input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Align with model feature order
input_encoded = input_encoded.reindex(columns=expected_encoded, fill_value=0)

# Scale numerical features
input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

# =========================
# Prediction
# =========================
if st.button("🔍 Predict Heart Disease"):
    prediction = model.predict(input_encoded)[0]

    if prediction == 0:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ No Strong Signs of Heart Disease")

st.caption("Developed by Sachin Chaudhary | © 2026 | Machine Learning Project")

