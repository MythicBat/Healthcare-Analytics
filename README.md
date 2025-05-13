# 💓 Healthcare-Analytics
This project aims to predict the likelihood of hospital readmission within 30 days for heart failure patients using clinical and
demographic data. It combines machine learning, Streamlit-based UI, and data visualization dashboards to assist healthcare professionals
in identifying high-risk patients and improving outcome.

---

## 📂 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Modeling Approach](#modeling-approach)
- [Sreamlit App Features](#streamlit-app-features)
- [How to Run](#how-to-run)
- [Insights](#insights)
- [Recommendations](#recommendations)

---

## 📌 Problem Statement
Heart failure patients are frequently readmitted to hospitals, costing time, money, and impacting care quality. This project
predicts which patients are at a risk of readmission based on their vitals, demographics, and lab results.

---

## 📊 Dataset
- **Source**: [Dataset Link](https://www.kaggle.com/datasets/programmer3/ghw-heart-failure-readmission-prediction-dataset?select=GHW_HeartFailure_Readmission.csv)
- **Target Variable**: `Readmission_30Days` (0 = No, 1 = Yes)
- **Features**: Age, Gender, Sodium, NT-proBNP, Length of stay, Creatinine, BP, Hemoglobin, etc.
- **Total**: 19 features used for training.

---

## 🧱 Project Structure
```
.
├── app.py                  # Streamlit app
├── GHW.py                  # Model training, EDA, and preprocessing
├── xgb_readmission_model.pkl
├── data/
│   └── GHW_HeartFailure_Readmission.csv
├── requirements.txt
└── README.md
```

---

## 🧠 Modeling Approach
- **Preprocessing**:
  - Label encoding for categorical data (Gender, Ethnicity, Discharge Type).
  - Missing values imputed using mean.
  - StandardScaler for numerical features.
- **Handling Imbalance**:
  - SMOTE oversampling applied to balance readmitted vs non-readmitted cases.
- **Model**:
  - XGBoost with hyperparameter tuning via GridSearchCV
  - Final mode saved as `xgb_readmission_model.pkl`
- **Evaluation**
  - AUC Score: `0.93`
  - Accuracy: `~89%`

---

## 🖥️ Streamlit App Features (`app.py`)
- Input patient's information via sidebar (age, vitals, etc)
- Predicts **Low** or **High** readmission risk with confidence score.
- Displays:
  - Color-coded bar chart: Patient vs Population vital values
  - Risk Summary Table: Identifies "Normal", "Moderate", "High" based on thresholds

---

## ▶️ How to Run

1. 🔧 Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   
---

## 🔍 Insights
- **Top Predictors**: NT-proBNP, Length of Stay, Creatinine, Sodium.
- Acheived **93%** AUC score using tuning XGBoost.
- App UI provides intuitive visuals and confidence-based prediction to support clinical decisions.

---

## 💡 Recommendations
- Deploy Streamlit app using Streamlit Cloud or GCP.
- Scale the dataset with real-world clinical sources.
- Integrate with hospital EMR (Electronic Medical Records).
- Add SHAP explainability for transparency. 
