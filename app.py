import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="log D Gas Diffusion in MOFs", layout="centered")

st.title("💻 Diffusion Coefficient Prediction for Gases in MOFs")

st.markdown("Select a gas type, enter the MOF features, and click **Predict** to get log D.")

# Gas selection
gas_type = st.selectbox("Select Gas", ["CO2", "O2", "N2", "CH4", "H2"])

# Feature input
feature_labels = [
    'PLD (Å)', 'LCD (Å)', 'LCD/PLD', 'Accessible Surface Area (m2/g)', 'Porosity',
    'Pore Volume (cm3/g)', 'Qst (kJ/mol)', 'Uptake at 1 bar (mol/kg)', 'log D0 (cm2/s)'
]

features = []
for label in feature_labels:
    val = st.number_input(label, format="%.6f")
    features.append(val)

# Prediction function
def predict_logD(features, gas_type):
    model_path = os.path.join("models", f"{gas_type.lower()}_model.pkl")
    model = joblib.load(model_path)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Predict button
if st.button("Predict"):
    if len(features) == 9:
        try:
            result = predict_logD(features, gas_type)
            st.success(f"log D for {gas_type} = {format(result, '.3g')}")
        except Exception as e:
            st.error("Prediction failed. Check model files.")
            st.text(str(e))
    else:
        st.warning("Please fill in all features.")
