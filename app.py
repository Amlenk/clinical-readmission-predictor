import os
os.environ['KERAS_BACKEND'] = 'torch'

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model

# Page config
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #F8FAFC;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 10px rgba(99, 102, 241, 0.3);
    }
    .stButton>button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.6);
    }
    .card {
        background-color: #1E293B;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }
    .card-title {
        color: #F8FAFC;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #475569;
        padding-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #6366F1;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #94A3B8;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🏥 Hospital Readmission Risk Predictor</h1>", unsafe_allow_html=True)

# Check if model and preprocessors exist
model_path = "best_model_bayesian.keras"
scaler_path = "scaler.joblib"
columns_path = "feature_columns.joblib"

if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(columns_path):
    st.error("Error: Trained model assets not found! Please wait for the Bayesian notebook execution to finish generating scaler.joblib, feature_columns.joblib, and best_model_bayesian.keras.")
    st.stop()

@st.cache_resource
def load_assets():
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    columns = joblib.load(columns_path)
    return model, scaler, columns

try:
    model, scaler, columns = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card"><div class="card-title">Patient Profile & Clinical Inputs</div>', unsafe_allow_html=True)
    
    # Sub-columns
    sub_col1, sub_col2 = st.columns(2)
    
    with sub_col1:
        st.subheader("Demographics & History")
        age = st.selectbox("Age Bracket", ['[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], index=3)
        time_in_hospital = st.slider("Time in Hospital (Days)", 1, 14, 4)
        n_outpatient = st.number_input("Number of Outpatient Visits (Previous Year)", 0, 40, 0)
        n_inpatient = st.number_input("Number of Inpatient Visits (Previous Year)", 0, 20, 0)
        n_emergency = st.number_input("Number of Emergency Visits (Previous Year)", 0, 70, 0)
        medical_specialty = st.selectbox("Medical Specialty of Admitting Physician", 
                                         ['Missing', 'Other', 'InternalMedicine', 'Family/GeneralPractice', 'Cardiology', 'Surgery', 'Emergency/Trauma'])
        
    with sub_col2:
        st.subheader("Clinical Procedures & Tests")
        n_lab_procedures = st.slider("Number of Lab Procedures", 1, 120, 45)
        n_procedures = st.slider("Number of Non-Lab Procedures", 0, 6, 1)
        n_medications = st.slider("Number of Medications Administered", 1, 80, 15)
        
        glucose_test = st.selectbox("Glucose Test Result", ['no', 'normal', 'high'])
        A1Ctest = st.selectbox("A1C Test Result", ['no', 'normal', 'high'])
        change = st.selectbox("Medication Change (Yes/No)", ['no', 'yes'])
        diabetes_med = st.selectbox("Diabetes Medication Prescribed (Yes/No)", ['yes', 'no'])
        
    st.subheader("Diagnostic Classification")
    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        diag_1 = st.selectbox("Primary Diagnosis (Diag 1)", ['Circulatory', 'Other', 'Injury', 'Digestive', 'Respiratory', 'Diabetes', 'Musculoskeletal', 'Missing'])
    with d_col2:
        diag_2 = st.selectbox("Secondary Diagnosis (Diag 2)", ['Respiratory', 'Other', 'Circulatory', 'Injury', 'Diabetes', 'Digestive', 'Musculoskeletal', 'Missing'])
    with d_col3:
        diag_3 = st.selectbox("Tertiary Diagnosis (Diag 3)", ['Other', 'Circulatory', 'Diabetes', 'Respiratory', 'Injury', 'Musculoskeletal', 'Digestive', 'Missing'])
        
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">Prediction Output</div>', unsafe_allow_html=True)
    
    predict_clicked = st.button("Analyze Readmission Risk", use_container_width=True)
    
    if predict_clicked:
        # 1. Map simple ordinal/categorical features
        age_map = {'[40-50)': 0, '[50-60)': 1, '[60-70)': 2, '[70-80)': 3, '[80-90)': 4, '[90-100)': 5}
        binary_map = {'no': 0, 'yes': 1}
        test_map = {'high': 0, 'no': 1, 'normal': 2}
        diabetes_med_map = {'no': 0, 'yes': 1}
        
        mapped_age = age_map[age]
        mapped_change = binary_map[change]
        mapped_diabetes_med = diabetes_med_map[diabetes_med]
        mapped_glucose = test_map[glucose_test]
        mapped_a1c = test_map[A1Ctest]
        
        # 2. Compute engineered features
        medications_per_day = n_medications / (time_in_hospital + 1)
        lab_tests_per_day = n_lab_procedures / (time_in_hospital + 1)
        healthcare_utilization = n_outpatient + n_inpatient + n_emergency
        stay_to_medication_ratio = time_in_hospital / (n_medications + 1)
        
        # 3. Construct input dictionary
        input_data = {
            'age': mapped_age,
            'time_in_hospital': time_in_hospital,
            'n_lab_procedures': n_lab_procedures,
            'n_procedures': n_procedures,
            'n_medications': n_medications,
            'n_outpatient': n_outpatient,
            'n_inpatient': n_inpatient,
            'n_emergency': n_emergency,
            'glucose_test': mapped_glucose,
            'A1Ctest': mapped_a1c,
            'change': mapped_change,
            'diabetes_med': mapped_diabetes_med,
            'medications_per_day': medications_per_day,
            'lab_tests_per_day': lab_tests_per_day,
            'healthcare_utilization': healthcare_utilization,
            'stay_to_medication_ratio': stay_to_medication_ratio
        }
        
        # 4. Handle dummy columns (medical_specialty, diag_1, diag_2, diag_3)
        # Initialize all dummy columns to 0
        for col in columns:
            if col not in input_data:
                input_data[col] = 0
                
        # Set selected categories to 1
        spec_col = f"medical_specialty_{medical_specialty}"
        d1_col = f"diag_1_{diag_1}"
        d2_col = f"diag_2_{diag_2}"
        d3_col = f"diag_3_{diag_3}"
        
        if spec_col in input_data:
            input_data[spec_col] = 1
        if d1_col in input_data:
            input_data[d1_col] = 1
        if d2_col in input_data:
            input_data[d2_col] = 1
        if d3_col in input_data:
            input_data[d3_col] = 1
            
        # 5. Convert to dataframe with correct column order
        df_input = pd.DataFrame([input_data])
        df_input = df_input[columns]
        
        # 6. Apply Standard Scaling to numerical columns
        numerical_columns = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 
                             'n_outpatient', 'n_inpatient', 'n_emergency', 'medications_per_day', 
                             'lab_tests_per_day', 'healthcare_utilization', 'stay_to_medication_ratio']
        df_input[numerical_columns] = scaler.transform(df_input[numerical_columns])
        
        # 7. Model prediction
        pred_prob = float(model.predict(df_input, verbose=0)[0][0])
        
        # Display Prediction
        st.markdown(f'<div class="metric-value">{pred_prob * 100:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Predicted Readmission Risk</div>', unsafe_allow_html=True)
        st.write("")
        
        # Risk Classification
        if pred_prob >= 0.55:
            st.error("🚨 HIGH RISK: Patient has a high risk of readmission within 30 days. Recommend standard discharge follow-up protocol and transition care plan.")
        elif pred_prob >= 0.45:
            st.warning("⚠️ MODERATE RISK: Patient has elevated readmission risk. Monitor transition care and verify medication changes are clearly explained.")
        else:
            st.success("✅ LOW RISK: Patient has low readmission risk. Standard discharge procedures are appropriate.")
            
        # Clinical Risk Factors explanation
        st.subheader("Key Risk Contributors")
        reasons = []
        if n_inpatient > 1:
            reasons.append("- **Prior Inpatient Visits**: High prior inpatient utilization is a strong indicator of chronicity and readmission.")
        if medications_per_day > 4.0:
            reasons.append("- **High Medication Intensity**: Large volume of administered medications suggests higher disease complexity.")
        if time_in_hospital > 7:
            reasons.append("- **Length of Stay**: Hospitalization > 7 days indicates severity of the episode.")
        if change == 'yes':
            reasons.append("- **Medication Changes**: Active adjustments in diabetes treatment during stay increase follow-up complexity.")
        if n_emergency > 0:
            reasons.append("- **Emergency History**: Previous emergency encounters strongly correlate with unstable outpatient status.")
            
        if reasons:
            for r in reasons:
                st.write(r)
        else:
            st.write("Patient profiles do not indicate any major historical or procedure-based risk flags.")
            
    else:
        st.info("Adjust the patient clinical parameters on the left and click 'Analyze Readmission Risk' to see predictions.")
        
    st.markdown('</div>', unsafe_allow_html=True)
