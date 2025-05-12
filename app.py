import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go

# Load model
model = joblib.load("xgb_readmission_model.pkl")

# Load the dataset
df = pd.read_csv("data/GHW_HeartFailure_Readmission.csv")

# Manual label encoding based on training LabelEncoder mappings
gender_map = {'Male': 1, 'Female': 0}
ethnicity_map = {'Other': 0, 'Hispanic': 1, 'White': 2, 'Asian': 3, 'Black': 4}
discharge_map = {'Rehab': 2, 'Home': 1, 'Expired': 0, 'Nursing Facility': 3}

# --- Page Config ---
st.set_page_config(page_title="Heart Failure Readmission Predictor", layout="centered")

# --- App Header ---
st.markdown("<h1 style='text-align: center; color: crimson;'>💓 Heart Failure Readmission Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter patient details to predict hospital readmission risk within 30 days.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("📝 Patient Information")

patient_id = st.sidebar.number_input("Patient ID", min_value=0, value=1000)
age = st.sidebar.slider("Age", 18, 100, 65)
gender = st.sidebar.selectbox("Gender", list(gender_map.keys()))
ethnicity = st.sidebar.selectbox("Ethnicity", list(ethnicity_map.keys()))
length_of_stay = st.sidebar.slider("Length of Stay (Days)", 1, 30, 5)
previous_admissions = st.sidebar.slider("Previous Admissions", 0, 10, 2)
discharge = st.sidebar.selectbox("Discharge Disposition", list(discharge_map.keys()))
pulse = st.sidebar.number_input("Pulse", 40, 180, value=75)
temperature = st.sidebar.number_input("Temperature (°C)", 35.0, 42.0, value=37.0, step=0.1)
heart_rate = st.sidebar.number_input("Heart Rate", 40, 180, value=80)
systolic_bp = st.sidebar.number_input("Systolic BP", 80, 200, value=120)
diastolic_bp = st.sidebar.number_input("Diastolic BP", 40, 130, value=80)
respiratory_rate = st.sidebar.number_input("Respiratory Rate", 10, 40, value=18)
bun = st.sidebar.number_input("BUN (mg/dL)", 1, 100, value=18)
creatinine = st.sidebar.number_input("Creatinine (mg/dL)", 0.1, 5.0, value=1.2)
sodium = st.sidebar.number_input("Sodium (mEq/L)", 120, 160, value=140)
hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", 5.0, 20.0, value=13.0)
nt_probnp = st.sidebar.number_input("NT-proBNP (pg/mL)", 0, 50000, value=2000)
ejection_fraction = st.sidebar.slider("Ejection Fraction (%)", 10, 70, 50)

predict_button = st.sidebar.button("🔍 Predict")

# --- Make Prediction ---
if predict_button:
    input_array = np.array([[
        patient_id,
        age,
        gender_map[gender],
        ethnicity_map[ethnicity],
        length_of_stay,
        previous_admissions,
        discharge_map[discharge],
        pulse,
        temperature,
        heart_rate,
        systolic_bp,
        diastolic_bp,
        respiratory_rate,
        bun,
        creatinine,
        sodium,
        hemoglobin,
        nt_probnp,
        ejection_fraction
    ]])

    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    result = "🔴 High Risk" if prediction == 1 else "🟢 Low Risk"
    risk_style = "color: red;" if prediction == 1 else "color: green;"

    st.markdown(f"""
        <div style='background-color: #f9f9f9; padding: 20px; border-radius: 12px; box-shadow: 0 0 10px rgba(0,0,0,0.1);'>
            <h3 style='{risk_style} text-align: center;'>{result}</h3>
            <p style='text-align: center;'>Model confidence: <b>{probability:.2%}</b></p>
        </div>
    """, unsafe_allow_html=True)

    # --- Patient vs Population Vitals --- #
    st.markdown("📊 Patient vs Population Comparision")

    # Define key vitals
    vital_col = ['Sodium', 'Creatinine', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Hemoglobin']
    patient_vals = [sodium, creatinine, systolic_bp, diastolic_bp, heart_rate, hemoglobin]
    mean_vals = df[vital_col].mean().tolist()

    colors = []
    for p, m in zip(patient_vals, mean_vals):
        diff = abs(p - m)
        if diff > 20:
            colors.append('red')
        elif diff > 10:
            colors.append('orange')
        else:
            colors.append('green')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = vital_col,
        y = patient_vals,
        name = 'Patient',
        marker_color = colors
    ))
    fig.add_trace(go.Bar(
        x = vital_col,
        y = mean_vals,
        name = 'Population Avg',
        marker_color = 'lightgray'
    ))
    fig.update_layout(barmode='group', title="Patient vs Population Vital Comparison")

    st.plotly_chart(fig, use_container_width=True)

    # Risk Summary Table
    st.markdown("🚨 Risk Summary Table")
    normal_ranges = {
        'Sodium': (135, 145),
        'Creatinine': (0.6, 1.3),
        'Systolic_BP': (90, 120),
        'Diastolic_BP': (60, 80),
        'Heart_Rate': (60, 100),
        'Hemoglobin': (12.0, 16.0)
    }

    summary = []
    for col, val in zip(vital_col, patient_vals):
        low, high = normal_ranges[col]
        if val < low or val > high:
            risk = '⚠️ High Risk'
        elif abs(val - (low + high)/2) > ((high - low)/4):
            risk = '🟠 Moderate'
        else:
            risk = '✅ Normal'
        summary.append((col, val, f"{low}-{high}", risk))

    # Display as Table
    summary_df = pd.DataFrame(summary, columns=["Vital", "Patient Value", "Normal Range", "Risk Level"])
    st.dataframe(summary_df)

else:
    st.info("⬅️ Enter patient info in the sidebar and click **Predict**.")

# --- Footer ---
st.markdown("---")
st.markdown("<small>Built with ❤️ using Streamlit and XGBoost</small>", unsafe_allow_html=True)