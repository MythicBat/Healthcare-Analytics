import numpy as np
import streamlit as st
import joblib

# Load model
model = joblib.load("xgb_readmission_model.pkl")

st.title("💓 Heart Failure Readmission Predictor")

# User Inputs
age = st.slider("Age", 18,100)
gender = st.selectbox("Gender", ["Male", "Female"])
bp = st.number_input("Blood Pressure", 80, 200)
sodium = st.number_input("Sodium Level", 120, 160)
creatinine = st.number_input("Creatinine Level", 0.1, 5.0)
prior_admission = st.slider("Number of Prior Admissions", 0, 10)
length_of_stay = st.slider("Length of Stay (Days)", 1, 30)

if st.button("Predict"):
    result = "🔴 High Risk" if prediction[0] == 1 else "🟢 Low Risk"
    st.success(f"Prediction: {result}")