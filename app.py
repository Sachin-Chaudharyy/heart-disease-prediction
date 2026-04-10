import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from fpdf import FPDF

st.title("❤️ Heart Disease Prediction")

try:
    with open("heart_disease_model.pkl","rb") as file:
        model = pickle.load(file)
    with open("scaler.pkl","rb") as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file not found. Please make sure 'heart_disease_model.pkl' is in the same folder.")
    st.stop() 
except pickle.UnpicklingError:
    st.error("❌ Failed to load the model. The file may be corrupted.")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error: {e}")
    st.stop()


categorical_cols = ['ChestPainType', 'RestingECG', 'ST_Slope']

numerical_cols = ['Age', 'Cholesterol', 'RestingBP', 'MaxHR', 'ST_Depression']

expected_encoded = model.feature_names_in_

st.set_page_config(page_title="Heart Disease Prediction")

dark_mode = st.checkbox("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
        <style>
        .stApp {
            background-color: #1a1c23;
            color: #e0e0e0;
        }
        [data-testid="stSidebar"] {
            background-color: #252832;
            color: #e0e0e0;
        }
        .stMarkdown, .stText {
            color: #ffffff !important;
        }
        label, .stTextInput label, .stNumberInput label, .stSelectbox label {
            color: #ffffff !important;
            font-weight: 500;
        }
        input, textarea, .stSelectbox div {
            background-color: #2c2f38 !important;
            color: #ffffff !important;
        }
        button, .stButton>button {
            background-color: #3b3f4b;
            color: #ffffff;
            border: 1px solid #4a4f5d;
        }
        button:hover, .stButton>button:hover {
            background-color: #4a4f5d;
        }
        </style>
        """, unsafe_allow_html=True
    )
else:
    st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff;
            color: #1a1c23;
        }
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        .stMarkdown, .stText {
            color: #1a1c23 !important;
        }
        label, .stTextInput label, .stNumberInput label, .stSelectbox label {
            color: #1a1c23 !important;
            font-weight: 500;
        }
        input, textarea, .stSelectbox div {
            background-color: #ffffff !important;
            color: #1a1c23 !important;
        }
        button, .stButton>button {
            background-color: #f0f2f6;
            color: #1a1c23;
            border: 1px solid #d1d5db;
        }
        </style>
        """, unsafe_allow_html=True
    )


st.write("""
This app predicts the likelihood of heart disease based on patient medical data.
""")
st.warning("""
⚠️ **Important:** This prediction is for educational purposes only and **does not replace professional medical advice**.  
Please consult a qualified doctor for a proper diagnosis and treatment.
""")
st.markdown("Developed by **Sachin Chaudhary** | Machine Learning Project | © 2026")
st.markdown("❤️ 💓 💊 🏥")
st.markdown("---")

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

st.sidebar.header("Patient Information")

Age = st.sidebar.number_input("Age (years)", 10, 100, 45)
Gender_display = st.sidebar.selectbox("Gender", list(gender_map.keys()))
ChestPainType_display = st.sidebar.selectbox("Chest Pain Type", list(chest_pain_map.keys()))
RestingBP = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
Cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 200)
FastingBS_display = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", list(fasting_bs_map.keys()))

RestingECG_display = st.sidebar.selectbox("Resting ECG Results", list(resting_ecg_map.keys()))
MaxHR = st.sidebar.number_input("Maximum Heart Rate (bpm)", 60, 220, 150)
ExerciseAngina_display = st.sidebar.selectbox("Exercise-Induced Angina", list(exercise_angina_map.keys()))
ST_Depression = st.sidebar.number_input("ST Depression (mm)", 0.0, 6.0, 1.0, step=0.1)
ST_Slope_display = st.sidebar.selectbox("ST Slope", list(st_slope_map.keys()))


Gender = gender_map[Gender_display]
ChestPainType = chest_pain_map[ChestPainType_display]
FastingBS = fasting_bs_map[FastingBS_display]
RestingECG = resting_ecg_map[RestingECG_display]
ExerciseAngina = exercise_angina_map[ExerciseAngina_display]
ST_Slope = st_slope_map[ST_Slope_display]


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
    "ST_Slope": ST_Slope
}


feature_names_map = {
    "Age": "Age",
    "Gender": "Gender (Male)",
    "RestingBP": "Blood Pressure",
    "Cholesterol": "Cholesterol Level",
    "MaxHR": "Max Heart Rate",
    "ST_Depression": "ST Depression",

    "ChestPainType_1": "Chest Pain: Atypical",
    "ChestPainType_2": "Chest Pain: Non-Anginal",
    "ChestPainType_3": "Chest Pain: Asymptomatic",

    "RestingECG_1": "ECG: ST-T Abnormality",
    "RestingECG_2": "ECG: Hypertrophy",

    "ST_Slope_1": "ST Slope: Flat",
    "ST_Slope_2": "ST Slope: Downsloping"
}

def analyze_parameters(data):
    issues = []

    if data["RestingBP"] > 130:
        issues.append("⚠️ Blood Pressure is HIGH")
    elif data["RestingBP"] < 90:
        issues.append("⚠️ Blood Pressure is LOW")

    if data["Cholesterol"] > 240:
        issues.append("⚠️ Cholesterol is HIGH")
    elif data["Cholesterol"] < 150:
        issues.append("⚠️ Cholesterol is LOW")

    if data["MaxHR"] < 90:
        issues.append("⚠️ Heart Rate is LOW")
    elif data["MaxHR"] > 180:
        issues.append("⚠️ Heart Rate is HIGH")

    if data["ST_Depression"] > 2:
        issues.append("⚠️ ST Depression is HIGH (possible abnormality)")

    if data["FastingBS"] == 1:
        issues.append("⚠️ Blood Sugar is HIGH")

    if data["ExerciseAngina"] == 1:
        issues.append("⚠️ Exercise-induced Angina present")

    return issues

input_df = pd.DataFrame([input_dict])

# One-hot encode categorical columns
input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Align with model feature order
input_encoded = input_encoded.reindex(columns=expected_encoded, fill_value=0)

# Scale numerical features
input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])


if st.button("🔍 Predict Heart Disease"):
    st.markdown("## 🧾 Patient Health Summary")
    recommendation = ""
    risk = model.predict_proba(input_encoded)[0][1]*100
    if risk >= 70:
        st.error(f"⚠️ High Risk ({risk:.1f}%)")
        st.markdown("🚨 Your current health indicators suggest a serious risk. Immediate medical attention is strongly advised.")
        recommendation = "Immediate consultation with a cardiologist is strongly recommended."
    elif risk >= 40:
        st.warning(f"⚠️ Moderate Risk ({risk:.1f}%)")
        st.markdown("⚠️ There are warning signs. Improving lifestyle and regular monitoring is important.")
        recommendation = "Adopt a healthier lifestyle and schedule regular medical checkups."
    else:
        st.success(f"✅ Low Risk ({risk:.1f}%)")
        st.markdown("✅ Your health indicators look stable. Keep maintaining your current lifestyle.")
        recommendation = "Maintain your current healthy lifestyle."
        
    st.info(f"💡 Recommendation: {recommendation}")
    st.progress(risk/100)
    issues = analyze_parameters(input_dict)

    st.markdown("### ⚠️ Health Parameter Analysis")

    if issues:
        for issue in issues:
            st.warning(issue)
    else:
        st.success("✅ All parameters are within normal range")

    with st.expander("📊 View Risk Distribution"):
            labels = ['No Heart Disease', 'Heart Disease']
            sizes = [100 - risk, risk]
            colors = ['#2a9d8f', '#e63946']

            fig, ax = plt.subplots(figsize=(3,3)) 
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
            ax.axis('equal')
            st.pyplot(fig, use_container_width=False)


    with st.expander("📉 See Risk Factors"):
        importance = getattr(model, "coef_", None)

        if importance is not None:
            importance = pd.Series(importance[0], index=input_encoded.columns)


            clean_data = {}

            for col, val in importance.items():

                # Chest Pain (only show selected one)
                if col.startswith("ChestPainType_"):
                    if input_encoded[col].values[0] == 1:
                        name = {
                            "ChestPainType_1": "Chest Pain: Atypical",
                            "ChestPainType_2": "Chest Pain: Non-Anginal",
                            "ChestPainType_3": "Chest Pain: Asymptomatic"
                        }.get(col, col)
                        clean_data[name] = val

                # ST Slope
                elif col.startswith("ST_Slope_"):
                    if input_encoded[col].values[0] == 1:
                        name = {
                            "ST_Slope_1": "ST Slope: Flat",
                            "ST_Slope_2": "ST Slope: Downsloping"
                        }.get(col, col)
                        clean_data[name] = val

                # ECG
                elif col.startswith("RestingECG_"):
                    if input_encoded[col].values[0] == 1:
                        name = {
                            "RestingECG_1": "ECG: ST-T Abnormality",
                            "RestingECG_2": "ECG: Hypertrophy"
                        }.get(col, col)
                        clean_data[name] = val

                # Normal features
                else:
                    name_map = {
                        "Age": "Age",
                        "Gender": "Gender (Male)",
                        "RestingBP": "Blood Pressure",
                        "Cholesterol": "Cholesterol",
                        "MaxHR": "Max Heart Rate",
                        "ST_Depression": "ST Depression",
                        "FastingBS": "High Blood Sugar",
                        "ExerciseAngina": "Exercise Angina"
                    }
                    clean_data[name_map.get(col, col)] = val

            clean_series = pd.Series(clean_data)

            clean_series = clean_series.sort_values()

            fig, ax = plt.subplots()

            ax.barh(clean_series.index, clean_series.values)

            ax.set_xlabel("Impact on Risk")
            ax.set_title("Key Risk Factors (Based on Your Input)")

            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)

        else:
            st.info("Feature importance not available for this model.")


    report_dict = {
    "Age": Age,
    "Gender": Gender_display,
    "Chest Pain Type": ChestPainType_display,
    "Resting Blood Pressure": RestingBP,
    "Cholesterol": Cholesterol,
    "Fasting Blood Sugar >120 mg/dl": FastingBS_display,
    "Resting ECG": RestingECG_display,
    "Max Heart Rate": MaxHR,
    "Exercise-Induced Angina": ExerciseAngina_display,
    "ST Depression": ST_Depression,
    "ST Slope": ST_Slope_display,
    "Predicted Risk (%)": f"{risk:.1f}%",
    "Recommendation": recommendation
    }
    st.markdown("### 📥 Download Report")

    report_df = pd.DataFrame([report_dict])
    csv = report_df.to_csv(index=False)

    col1, col2 = st.columns(2)

    col1.download_button(
        "📥 CSV Report",
        csv,
        file_name="heart_report.csv"
    )

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Heart Disease Report", ln=True, align="C")

    pdf.set_font("Arial",'',12)
    pdf.ln(10)

    for key, value in report_dict.items():
        pdf.multi_cell(0, 8, f"{key}: {value}")

    pdf.ln(5)
    pdf_output = pdf.output(dest='S').encode('latin-1')
    col2.download_button("📥 Download Report (PDF)", pdf_output, file_name="heart_report.pdf")


st.caption("🔍 This result is an estimate based on the provided inputs.")
st.caption("Developed by Sachin Chaudhary | © 2026 | Machine Learning Project")
