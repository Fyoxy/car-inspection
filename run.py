import streamlit as st
import pandas as pd
from pycaret.classification import *
import joblib

# Load encodings and models once
encoding_mappings = joblib.load('encoding_mappings.pkl')
models_to_use = {
    'Logistic Regression': load_model('LogisticRegression_model'),
    'Decision Tree': load_model('DecisionTreeClassifier_model'),
    'Random Forest': load_model('RandomForestClassifier_model'),
    'Naive Bayes': load_model('GaussianNB_model'),
    'XGBoost': load_model('XGBClassifier_model')
}

# Encoding function
def encode_input(input_data):
    for column, mapping in encoding_mappings.items():
        input_data[column] = {v: k for k, v in mapping.items()}.get(input_data[column], -1)
    return input_data

# Map months to names
months = {
    "Jaanuar": 1, "Veebruar": 2, "Märts": 3, "Aprill": 4,
    "Mai": 5, "Juuni": 6, "Juuli": 7, "August": 8,
    "September": 9, "Oktoober": 10, "November": 11, "Detsember": 12
}

# App Title and Description
st.title("Auto ülevaatuse tulemuse prognoosimine")
st.write("Prognoosimine toimub transpordiameti ülevaatusandmetel aastast 2013 kuni 2023. Sisestage oma sõiduki andmed allpool.")

# Input fields in columns for better layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        single_case = {
            'vehicle_age': st.number_input("Auto vanus (aastates)", value=10, min_value=0, max_value=150),
            'TEHNOYLEVAATUSPUNKT': st.selectbox("Tehnoülevaatuspunkt", list(encoding_mappings['TEHNOYLEVAATUSPUNKT'].values())),
            'YLEVAATUSLIIK': st.selectbox("Ülevaatusliik", list(encoding_mappings['YLEVAATUSLIIK'].values())),
            'YV_KUUPAEV_MONTH': months[st.selectbox("Ülevaatuse planeeritav kuu", list(months.keys()))]
        }
    with col2:
        single_case.update({
            'MARK': st.selectbox("Auto Mark", list(encoding_mappings['MARK'].values())),
            'MUDEL': st.selectbox("Auto Mudel", list(encoding_mappings['MUDEL'].values())),
            'KATEGOORIA': st.selectbox("Auto Kategooria", list(encoding_mappings['KATEGOORIA'].values())),
            'KERETYYP': st.selectbox("Auto Keretüüp", list(encoding_mappings['KERETYYP'].values()))
        })
    
    # Submit button
    submitted = st.form_submit_button("Prognoosi")

# Prediction logic on button click
if submitted:
    st.subheader("Mudelite prognoosid")
    st.write("Allpool on iga mudeli prognoos auto ülevaatustulemuse kohta:")

    # Encode the input data for model compatibility
    encoded_input = encode_input(single_case)
    input_df = pd.DataFrame([encoded_input])

    korras_count = 0  # Count how many models predict "KORRAS"

    # Display predictions with color coding
    for model_name, model in models_to_use.items():
        prediction = predict_model(model, data=input_df)

        if 'prediction_label' in prediction.columns:
            result = prediction['prediction_label'].iloc[0]
            # Add color based on the result
            color = "green" if result == "KORRAS" else "red"
            if result == "KORRAS":
                korras_count += 1
            st.markdown(f"**{model_name}:** :{color}[{result}]")
        else:
            st.write(f"{model_name}: Tulemuse leidmine ebaõnnestus.")

    # Display summary of "KORRAS" results
    st.write(f"**KORRAS** tulemusi kokku: {korras_count} / {len(models_to_use)}")
