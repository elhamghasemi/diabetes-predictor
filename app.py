
import streamlit as st
import joblib
import numpy as np

# Load model and feature names
model = joblib.load('diabetes_best_model_cv.pkl')
features = joblib.load('best_feature_names.pkl')

# Page config
st.set_page_config(
    page_title="Diabetes Progression Predictor",
    page_icon="Chart",
    layout="centered"
)

st.title("Diabetes Disease Progression Predictor")
st.markdown("### One-year disease progression risk based on 3 key biomarkers")

st.info("This model uses only the 3 most important factors identified by machine learning")

# Sidebar inputs with real names and proper ranges
st.sidebar.header("Input Patient Data (standardized values)")

inputs = []
for feat in features:
    if feat == "bmi":
        label = "BMI (Body Mass Index)"
    elif feat == "bp":
        label = "Average Blood Pressure"
    else:  # s5
        label = "s5 - Serum Measurement (lt-globulin related)"
        
    val = st.sidebar.slider(
        label,
        min_value=-0.15,
        max_value=0.15,
        value=0.0,
        step=0.001,
        format="%.4f",
        help=f"Standardized value for {feat}"
    )
    inputs.append(val)

if st.sidebar.button("Predict Progression", type="primary"):
    # Prediction
    input_array = np.array([inputs])
    prediction = model.predict(input_array)[0]
    
    st.success(f"Predicted Disease Progression Score: **{prediction:.1f}**")
    
    # Risk interpretation
    if prediction > 200:
        st.error("High Risk – Rapid disease progression expected")
    elif prediction > 150:
        st.warning("Moderate to High Risk")
    elif prediction > 100:
        st.info("Moderate Risk")
    else:
        st.success("Lower Risk – Slower progression expected")
    
    st.balloons()

# Footer
st.markdown("---")
st.caption("Model: Ridge Regression | Features: s5, bmi, bp | CV R² ≈ 0.48")
