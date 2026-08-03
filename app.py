import streamlit as st 
import numpy as np
import joblib
import os

st.set_page_config(page_title="log D Gas Diffusion in MOFs", layout="centered")

st.title("💻 Diffusion Coefficient Prediction for Gases in MOFs")
st.markdown("Select a gas type, enter the MOF features, and click **Predict** to get log D.")

gas_type = st.selectbox("Select Gas", ["CO₂", "O₂", "N₂", "CH₄", "H₂", "Universal"])

if gas_type == "Universal":
    feature_labels = [
        'PLD (Å)',
        'LCD (Å)',
        'Accessible Surface Area (m²/g)',
        'Porosity',
        'Pore Volume (cm³/g)',
        'σ (Å)',
        'M.W. (g/mol)',
        'Qₛₜ (kJ/mol)',
        'Uptake 1 bar (mol/kg)',
        'log(D₀) (cm²/s)'
    ]
elif gas_type in ["CH₄", "H₂"]:
    feature_labels = [
        'PLD (Å)',
        'LCD (Å)',
        'Accessible Surface Area (m²/g)',
        'Porosity',
        'Pore Volume (cm³/g)',
        'Qₛₜ (kJ/mol)',
        'Uptake at 1 bar (mol/kg)',
        'log D₀ (cm²/s)'
    ]
else:
    feature_labels = [
        'PLD (Å)',
        'LCD (Å)',
        'LCD/PLD',
        'Accessible Surface Area (m²/g)',
        'Porosity',
        'Pore Volume (cm³/g)',
        'Qₛₜ (kJ/mol)',
        'Uptake at 1 bar (mol/kg)',
        'log D₀ (cm²/s)'
    ]

# Feature input
features = []
for label in feature_labels:
    val = st.number_input(label, format="%.6f")
    features.append(val)

# Prediction function
def predict_logD(features, gas_type):
    # model dosya adı yine orijinal gaz isimleriyle uyumlu olmalı
    model_file = gas_type.lower().replace("₂", "2").replace("₄", "4")
    model_path = os.path.join("models", f"{model_file}_model.pkl")
    model = joblib.load(model_path)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Predict button
if st.button("Predict"):
    expected_length = len(feature_labels)
    if len(features) == expected_length:
        try:
            result = predict_logD(features, gas_type)
            st.success(f"log D for {gas_type} = {format(result, '.3g')}")
        except Exception as e:
            st.error("Prediction failed. Check model files.")
            st.text(str(e))
    else:
        st.warning(f"Please fill in all {expected_length} features.")
