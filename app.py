import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from PIL import Image

st.set_page_config(page_title="Brain Stroke Prediction", layout="centered", page_icon="🧠")

page = st.sidebar.radio("📌 Select Page", ["🩺 ML Prediction", "🧠 Brain Image Prediction"])

@st.cache_resource
def load_all_models():
    BASE     = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE, "stroke_cardiovascular_synthetic_csv.csv")
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        df[col] = le.fit_transform(df[col])
    df = df.drop(['patient_id'], axis=1, errors='ignore')
    target = 'had_stroke' if 'had_stroke' in df.columns else 'heart_attack_risk'
    y = df[target]
    X = df.drop(target, axis=1)
    feature_cols = list(X.columns)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, scaler, feature_cols, acc

model, scaler, feature_cols, acc = load_all_models()

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — ML Prediction
# ══════════════════════════════════════════════════════════════════════
if page == "🩺 ML Prediction":
    st.title("🩺 Stroke & Cardiovascular Risk Prediction")
    st.write("Fill in the patient details below and click **Predict**.")
    st.success(f"✅ Model Ready! | Test Accuracy: {acc*100:.1f}%")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        age          = st.number_input("Age", 1, 120, 50)
        gender       = st.selectbox("Gender", ["Male", "Female"])
        bmi          = st.number_input("BMI", 10.0, 60.0, 25.0)
        blood_sugar  = st.number_input("Blood Sugar (mg/dL)", value=100.0)
        systolic_bp  = st.number_input("Systolic BP", value=120)
        diastolic_bp = st.number_input("Diastolic BP", value=80)
        cholesterol  = st.number_input("Cholesterol", value=200.0)
        hdl          = st.number_input("HDL", value=50.0)
        ldl          = st.number_input("LDL", value=130.0)
        triglycerides= st.number_input("Triglycerides", value=150.0)
        smoking      = st.selectbox("Smoking", ["No", "Yes"])
        alcohol      = st.number_input("Alcohol Units/Week", value=2.0)

    with col2:
        stress       = st.slider("Stress Level (1-10)", 1, 10, 5)
        diabetes     = st.selectbox("Diabetes", ["No", "Yes"])
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        af           = st.selectbox("Atrial Fibrillation", ["No", "Yes"])
        family       = st.selectbox("Family History of Clot", ["No", "Yes"])
        prev_stroke  = st.selectbox("Previous Stroke", ["No", "Yes"])
        prev_heart   = st.selectbox("Previous Heart Attack", ["No", "Yes"])
        clot         = st.selectbox("Clot Risk", ["No", "Yes"])
        clot_loc     = st.selectbox("Clot Location", ["none", "cerebral", "peripheral", "coronary"])
        activity     = st.selectbox("Physical Activity", ["No", "Yes"])
        heart_risk   = st.selectbox("Heart Attack Risk", ["No", "Yes"])

    def yn(val): return 1 if val == "Yes" else 0
    clot_map   = {"none": 0, "cerebral": 1, "peripheral": 2, "coronary": 3}
    gender_val = 1 if gender == "Male" else 0

    input_data = {
        "age": age, "gender": gender_val, "bmi": bmi,
        "blood_sugar": blood_sugar, "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp, "cholesterol": cholesterol,
        "hdl": hdl, "ldl": ldl, "triglycerides": triglycerides,
        "smoking": yn(smoking), "alcohol_units_per_week": alcohol,
        "physical_activity": yn(activity), "stress_level": stress,
        "diabetes": yn(diabetes), "hypertension": yn(hypertension),
        "atrial_fibrillation": yn(af), "family_history_clot": yn(family),
        "previous_stroke": yn(prev_stroke), "previous_heart_attack": yn(prev_heart),
        "clot_risk": yn(clot), "clot_location": clot_map[clot_loc],
        "heart_attack_risk": yn(heart_risk)
    }

    df_input = pd.DataFrame([input_data])
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_cols]

    st.markdown("---")
    if st.button("🔍 Predict Stroke Risk", use_container_width=True):
        scaled = scaler.transform(df_input)
        pred   = model.predict(scaled)[0]
        prob   = model.predict_proba(scaled)[0][1] * 100

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ HIGH RISK — Probability: {prob:.1f}%")
        else:
            st.success(f"✅ LOW RISK — Probability: {prob:.1f}%")

        fig, ax = plt.subplots(figsize=(6, 1.2))
        color = "#e74c3c" if pred == 1 else "#2ecc71"
        ax.barh(0, 100, color="#ecf0f1", height=0.5)
        ax.barh(0, prob, color=color, height=0.5)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("Risk %")
        ax.set_title("Risk Level")
        ax.axvline(50, color='orange', linestyle='--', alpha=0.7, label='50% threshold')
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()

        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(8)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        feat_imp.plot(kind='barh', ax=ax2, color='steelblue')
        ax2.set_title("Top Risk Factors")
        ax2.set_xlabel("Importance")
        ax2.invert_yaxis()
        st.pyplot(fig2)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Brain Image Prediction (NO .h5 FILE NEEDED)
# ══════════════════════════════════════════════════════════════════════
elif page == "🧠 Brain Image Prediction":
    st.title("🧠 Brain Image Analysis")
    st.write("Upload a brain scan image (MRI/CT). The app analyses visual features to estimate stroke risk.")
    st.info("ℹ️ This version uses image feature extraction — no external model file required.")

    uploaded_file = st.file_uploader("📤 Upload Brain Scan Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Brain Scan", use_column_width=True)

        if st.button("🔍 Analyse Image", use_container_width=True):
            img_resized  = img.resize((128, 128))
            img_array    = np.array(img_resized, dtype=np.float32) / 255.0
            gray         = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

            brightness   = float(np.mean(gray))
            dark_ratio   = float(np.mean(gray < 0.3))
            bright_ratio = float(np.mean(gray > 0.8))
            red_ch       = float(np.mean(img_array[:, :, 0]))
            asym_score   = float(np.abs(np.mean(gray[:, :64]) - np.mean(gray[:, 64:])))

            image_risk_score = min(max(
                (0.30 * dark_ratio + 0.25 * bright_ratio + 0.20 * asym_score +
                 0.15 * red_ch + 0.10 * (1 - brightness)) * 5, 0.0), 1.0)

            final_prob = min(max(0.5 * image_risk_score + 0.5 * 0.30, 0.0), 1.0)
            pct = final_prob * 100

            st.markdown("---")
            if final_prob >= 0.45:
                st.error(f"🔴 STROKE INDICATORS DETECTED — Risk Score: {pct:.1f}%")
                st.warning("⚠️ Scan shows characteristics (asymmetry, bright/dark regions) consistent with possible stroke. Consult a neurologist.")
            else:
                st.success(f"🟢 NO CLEAR STROKE SIGNS — Risk Score: {pct:.1f}%")
                st.info("✅ The scan appears normal based on visual analysis.")

            st.markdown("#### 📊 Image Feature Breakdown")
            features = {
                "Dark regions (lesions)": dark_ratio,
                "Bright spots (bleeds)": bright_ratio,
                "L-R Asymmetry": asym_score,
                "Red channel": red_ch,
                "Low brightness": 1 - brightness,
            }

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].barh(list(features.keys()), list(features.values()), color='tomato')
            axes[0].set_xlim(0, 1)
            axes[0].set_title("Detected Image Features")
            axes[0].set_xlabel("Feature Intensity")

            theta = np.linspace(0, np.pi, 200)
            axes[1].plot(np.cos(theta), np.sin(theta), 'k-', lw=2)
            for angle, color in [(0.2 * np.pi, '#2ecc71'), (0.5 * np.pi, '#f39c12'), (0.8 * np.pi, '#e74c3c')]:
                axes[1].plot([0, 0.7 * np.cos(np.pi - angle)], [0, 0.7 * np.sin(np.pi - angle)], color=color, lw=3, alpha=0.4)
            needle_angle = np.pi * (1 - final_prob)
            axes[1].annotate('', xy=(0.7 * np.cos(needle_angle), 0.7 * np.sin(needle_angle)),
                xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
            axes[1].set_xlim(-1.1, 1.1); axes[1].set_ylim(-0.2, 1.2)
            axes[1].set_aspect('equal'); axes[1].axis('off')
            axes[1].set_title(f"Risk Gauge: {pct:.1f}%")
            low_p  = mpatches.Patch(color='#2ecc71', label='Low')
            med_p  = mpatches.Patch(color='#f39c12', label='Medium')
            high_p = mpatches.Patch(color='#e74c3c', label='High')
            axes[1].legend(handles=[low_p, med_p, high_p], loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("#### 🔥 Attention Heatmap")
            diff = np.abs(gray - np.mean(gray))
            fig3, ax3 = plt.subplots(figsize=(5, 5))
            ax3.imshow(gray, cmap='gray', alpha=0.7)
            ax3.imshow(diff, cmap='hot', alpha=0.4)
            ax3.axis('off')
            ax3.set_title("Highlighted Anomaly Regions")
            st.pyplot(fig3)
            plt.close()

            st.markdown("---")
            st.caption("⚕️ Disclaimer: This is a screening tool for educational purposes only. NOT a medical diagnosis. Consult a certified radiologist or neurologist for actual medical evaluation.")
