import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

st.set_page_config(page_title="Stroke Prediction App", layout="centered")

st.title("🩺 Stroke & Cardiovascular Risk Prediction")
st.write("Fill patient details and click Predict.")

@st.cache_resource
def train_model():
    BASE = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE, "stroke_cardiovascular_synthetic_csv.csv")
    df = pd.read_csv(csv_path)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        df[col] = le.fit_transform(df[col])

    df = df.drop(['patient_id'], axis=1, errors='ignore')

    if 'had_stroke' in df.columns:
        y = df['had_stroke']
        X = df.drop('had_stroke', axis=1)
    else:
        y = df['heart_attack_risk']
        X = df.drop('heart_attack_risk', axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, list(X.columns)

model, scaler, feature_cols = train_model()

st.success("✅ Model Ready!")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    blood_sugar = st.number_input("Blood Sugar", 100.0)
    systolic_bp = st.number_input("Systolic BP", 120)
    diastolic_bp = st.number_input("Diastolic BP", 80)
    cholesterol = st.number_input("Cholesterol", 200.0)
    hdl = st.number_input("HDL", 50.0)
    ldl = st.number_input("LDL", 130.0)
    triglycerides = st.number_input("Triglycerides", 150.0)
    smoking = st.selectbox("Smoking", [0, 1])
    alcohol = st.number_input("Alcohol Units", 2.0)

with col2:
    activity = st.selectbox("Physical Activity", [0, 1])
    stress = st.slider("Stress Level", 1, 10, 5)
    diabetes = st.selectbox("Diabetes", [0, 1])
    hypertension = st.selectbox("Hypertension", [0, 1])
    af = st.selectbox("Atrial Fibrillation", [0, 1])
    family = st.selectbox("Family History", [0, 1])
    prev_stroke = st.selectbox("Previous Stroke", [0, 1])
    prev_heart = st.selectbox("Previous Heart Attack", [0, 1])
    clot = st.selectbox("Clot Risk", [0, 1])
    clot_loc = st.selectbox("Clot Location", ["none", "cerebral", "peripheral", "coronary"])
    heart_risk = st.selectbox("Heart Attack Risk", [0, 1])

clot_map = {"none": 0, "cerebral": 1, "peripheral": 2, "coronary": 3}
gender_val = 1 if gender == "Male" else 0

input_data = {
    "age": age, "gender": gender_val, "bmi": bmi,
    "blood_sugar": blood_sugar, "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp, "cholesterol": cholesterol,
    "hdl": hdl, "ldl": ldl, "triglycerides": triglycerides,
    "smoking": smoking, "alcohol_units_per_week": alcohol,
    "physical_activity": activity, "stress_level": stress,
    "diabetes": diabetes, "hypertension": hypertension,
    "atrial_fibrillation": af, "family_history_clot": family,
    "previous_stroke": prev_stroke, "previous_heart_attack": prev_heart,
    "clot_risk": clot, "clot_location": clot_map[clot_loc],
    "heart_attack_risk": heart_risk
}

df_input = pd.DataFrame([input_data])

for col in feature_cols:
    if col not in df_input.columns:
        df_input[col] = 0

df_input = df_input[feature_cols]

if st.button("🔍 Predict"):
    scaled = scaler.transform(df_input)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1] * 100

    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ HIGH RISK — {prob:.2f}%")
    else:
        st.success(f"✅ LOW RISK — {prob:.2f}%")
brain_model.h5
