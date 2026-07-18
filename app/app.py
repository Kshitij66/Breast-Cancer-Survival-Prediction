import streamlit as st
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predictor import predict
from src.utils import get_sample
from src.explain import plot_global_shap, plot_local_shap

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🩺 Breast Cancer Prediction System")

st.markdown("""
This application predicts whether a breast tumor is **Benign** or **Malignant**
using a trained **XGBoost Machine Learning model**.
""")

st.success("✅ Model Loaded Successfully")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Patient Information")

patient_name = st.sidebar.text_input("Patient Name")
patient_id = st.sidebar.text_input("Patient ID")
age = st.sidebar.number_input("Age", 1, 120, 40)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
doctor = st.sidebar.text_input("Doctor Name")

# -----------------------------
# Input Features
# -----------------------------
sample = get_sample()

st.header("Tumor Measurements")

for column in sample.columns:
    sample[column] = st.number_input(
        column,
        value=float(sample[column].iloc[0]),
        format="%.5f"
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):

    prediction, probability = predict(sample)

    prediction = int(prediction[0])

    # Current model mapping
    malignant_prob = float(probability[0][0] * 100)
    benign_prob = float(probability[0][1] * 100)

    confidence = max(malignant_prob, benign_prob)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if prediction == 0:

            st.metric(
                label="Diagnosis",
                value="🔴 Malignant"
            )

            st.error("⚠️ High Risk Tumor Detected")

        else:

            st.metric(
                label="Diagnosis",
                value="🟢 Benign"
            )

            st.success("✅ Benign Tumor")

    with col2:

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2f}%"
        )

    st.subheader("Prediction Confidence")

    st.write(f"🔴 Malignant : {malignant_prob:.2f}%")
    st.progress(float(malignant_prob / 100))

    st.write(f"🟢 Benign : {benign_prob:.2f}%")
    st.progress(float(benign_prob / 100))

    st.divider()

    st.subheader("Patient Summary")

    st.write(f"**Patient Name:** {patient_name}")
    st.write(f"**Patient ID:** {patient_id}")
    st.write(f"**Age:** {age}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Doctor:** {doctor}")
    st.divider()

    st.subheader("Model Explanation (SHAP)")

    tab1, tab2 = st.tabs([
    "Global Importance",
    "Patient Explanation"
    ])

    with tab1:

      fig = plot_global_shap(sample)
      st.pyplot(fig)

    with tab2:

      fig = plot_local_shap(sample)
      st.pyplot(fig)