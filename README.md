# Heart Disease Prediction App

This is a Machine Learning web application built using Streamlit that predicts the likelihood of heart disease based on patient medical data.

# Model Performance
- Logistic Regression Accuracy: 88.59%
- Random Forest Accuracy: 89.67%
- XGBoost Accuracy: 87.50%

# Features
- User-friendly UI with dark mode
- Real-time heart disease risk prediction
- Risk categorization (Low / Moderate / High)
- Visualization of prediction results
- Feature importance display
- Downloadable reports (CSV & PDF)

# Machine Learning
- Models used:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Final deployed model: Logistic Regression
- Data preprocessing:
  - One-hot encoding
  - Feature scaling using StandardScaler

# Dataset Download Link
- https://www.kaggle.com/datasets/eishkaran/heart-disease
- or you can simply download from my repository 

# Project Structure
├── app.py
├── model.ipynb
├── heart_disease_model.pkl
├── scaler.pkl
├── Dataset.csv
├── requirements.txt
└── README.md


# How to Run
1. Install dependencies:
    pip install -r requirements.txt

2. Run the app:
    streamlit run app.py


# Disclaimer
This project is for educational purposes only and should not be used for medical diagnosis.

# Author
Sachin Chaudhary
