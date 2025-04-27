import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

model = joblib.load('svr_model.pkl')  # Assurez-vous que votre modèle est un modèle de prédiction de la LOS
scaler = joblib.load('scaler.pkl')  # Charger le scaler sauvegardé
# Mappage des codes ICD9 avec les catégories
icd9_code_categories = {
    '40291': 'Cardiovascular Diseases',
    '41071': 'Cardiovascular Diseases',
    '41041': 'Cardiovascular Diseases',
    '42821': 'Cardiovascular Diseases',
    '4280': 'Cardiovascular Diseases',
    '43491': 'Cardiovascular Diseases',
    '486': 'Cardiovascular Diseases',
    '42781': 'Cardiovascular Diseases',
    '42741': 'Cardiovascular Diseases',
    '41401': 'Cardiovascular Diseases',
    '5715': 'Liver Diseases',
    '5770': 'Liver Diseases',
    '5722': 'Liver Diseases',
    '5750': 'Liver Diseases',
    '51881': 'Respiratory Disorders',
    '49322': 'Respiratory Disorders',
    '49121': 'Respiratory Disorders',
    '1508': 'Gastrointestinal Diseases',
    '53190': 'Gastrointestinal Diseases',
    '56985': 'Gastrointestinal Diseases',
    '1541': 'Gastrointestinal Diseases',
    '1124': 'Infectious and Parasitic Diseases',
    '389': 'Infectious and Parasitic Diseases',
    '7907': 'Infectious and Parasitic Diseases',
    '383': 'Infectious and Parasitic Diseases',
    '388': 'Infectious and Parasitic Diseases',
    '380': 'Infectious and Parasitic Diseases',
    '99731': 'Infectious and Parasitic Diseases',
    '431': 'Neurological and Nervous System Diseases',
    '80375': 'Neurological and Nervous System Diseases',
    '85221': 'Neurological and Nervous System Diseases',
    '85225': 'Neurological and Nervous System Diseases',
    '85206': 'Neurological and Nervous System Diseases',
    '20510': 'Neurological and Nervous System Diseases',
    '1983': 'Neurological and Nervous System Diseases',
    'V600': 'Others',
    'V5861': 'Others',
    'V0254': 'Others',
    '81249': 'Others',
    '71615': 'Others',
    '99667': 'Others',
    '5990': 'Others',
    '2859': 'Others',
    '7821': 'Symptoms',
    '53084': 'Gastrointestinal Disorders',
    '412': 'Cardiopathies',
    '1961': 'Tumors',
    '20280': 'Tumors',
    '41001': 'Cardiopathies',
    '1541': 'Gastrointestinal Diseases',
    '2511': 'Diabetes',
    '99591': 'Toxic and Infectious',
    '42823': 'Cardiopathies',
    '5761': 'Liver Disorders',
    '80601': 'Fractures',
    '543': 'Gastrointestinal Disorders',
    '9693': 'Injuries',
    '43411': 'Cerebral Accidents',
    '4588': 'Circulatory Disorders',
    '49121': 'Respiratory Disorders',
}

# Initialisation des encodeurs
label_encoder_admission_type = LabelEncoder()
label_encoder_insurance = LabelEncoder()
label_encoder_first_careunit = LabelEncoder()
label_encoder_last_careunit = LabelEncoder()
label_encoder_expire_flag = LabelEncoder()
label_encoder_icd9_code = LabelEncoder()

# Ajustement des encodeurs avec les valeurs possibles
label_encoder_admission_type.fit(['EMERGENCY', 'ELECTIVE', 'URGENT'])
label_encoder_insurance.fit(['Medicare', 'Private', 'Medicaid', 'Government'])
label_encoder_first_careunit.fit(['MICU', 'SICU', 'TSICU', 'CSRU', 'CCU'])
label_encoder_last_careunit.fit(['MICU', 'SICU', 'TSICU', 'CSRU', 'CCU'])
label_encoder_expire_flag.fit([1, 0])
label_encoder_icd9_code.fit(['7821', '5715', '53084', '51881', '40291', '389', '5770', '412', '431', '1508', '53190', 
                             '1961', '80375', '41071', '1124', '570', '43491', '56985', '4019', '3811', '85221', '8830',
                             '41041', '1890', '49322', '20280', 'V600', '1541', '2511', '42821', '99591', '81249', 
                             '42781', '1510', '3842', '4280', '42741', '1983', '41001', 'V5861', '486', '7907', '71615', 
                             '99667', '42823', '383', '5761', '80601', '81201', '543', '388', '9693', '5990', '99731', 
                             '85225', '20510', '1628', '380', '80125', '2859', 'V0254', '5722', '85206', '80501', '5750', 
                             '43411', '4588', '49121', '41401'])

# Fonction pour gérer les valeurs inconnues
def encode_with_unknown(encoder, data):
    try:
        return encoder.transform(data)
    except ValueError as e:
        print(f"Erreur d'encodage pour '{data}': {e}")
        # Retourner une valeur par défaut pour les valeurs inconnues
        return np.array([-1] * len(data))  # -1 comme valeur par défaut

def normalize_los(predicted_los):
    # Assurez-vous que la prédiction est positive avant d'appliquer log
    if predicted_los > 0:
        return np.log(predicted_los)
    else:
        return 0  # Gérer les valeurs négatives ou zéro si nécessaire

# # Définition de la barre de navigation
# def navigation():
#     st.sidebar.title("Menu")
#     page = st.sidebar.radio("", ["Home Page", "LOS Prediction Page"])
#     return page

# # Page d'accueil
# def home_page():
#     st.title("Welcome to the Length of Stay Prediction App")
#     st.subheader("We predict the length of stay in the hospital with our model")

# Page de prédiction de la LOS
def prediction_page():
    st.title("Length of Stay (LOS) Prediction")
    # Entrées utilisateur pour chaque feature
    admission_type = st.selectbox("Admission Type", options=['EMERGENCY', 'ELECTIVE', 'URGENT'])
    insurance = st.selectbox("Insurance Type", options=['Medicare', 'Private', 'Medicaid', 'Government'])
    first_careunit = st.selectbox("First Care Unit", options=['MICU', 'SICU', 'TSICU', 'CSRU', 'CCU'])
    last_careunit = st.selectbox("Last Care Unit", options=['MICU', 'SICU', 'TSICU', 'CSRU', 'CCU'])
    expire_flag = st.selectbox("Deceased at Hospital?", options=[1, 0])

    # Entrée de code ICD9 par l'utilisateur
    icd9_code_input = st.text_input("Enter an ICD9 code (e.g., 40291)")

    # Vérification du code ICD9
    if icd9_code_input:
        # Find the corresponding diagnosis for the code
        category = icd9_code_categories.get(icd9_code_input)
        
        if category:
            st.write(f"The ICD9 code {icd9_code_input} belongs to the category: {category}")
        else:
            st.write("The ICD9 code is not valid or not in the database.")
            
    icu_stay_duration = st.number_input("ICU Stay Duration (in days)", min_value=0, step=1)
    ed_waiting_time = st.number_input("Emergency Department Waiting Time (in minutes)", min_value=0, step=1)
    duration_treatment = st.number_input("Treatment Duration (in days)", min_value=0, step=1)
    age = st.number_input("Patient Age", min_value=0, step=1)

    admission_type_encoded = encode_with_unknown(label_encoder_admission_type, [admission_type])[0]
    insurance_encoded = encode_with_unknown(label_encoder_insurance, [insurance])[0]
    first_careunit_encoded = encode_with_unknown(label_encoder_first_careunit, [first_careunit])[0]
    last_careunit_encoded = encode_with_unknown(label_encoder_last_careunit, [last_careunit])[0]
    expire_flag_encoded = encode_with_unknown(label_encoder_expire_flag, [expire_flag])[0]
    icd9_code_input_encoded = encode_with_unknown(label_encoder_icd9_code, [icd9_code_input])[0]

    features = [
        admission_type_encoded, insurance_encoded, first_careunit_encoded, last_careunit_encoded,
        expire_flag_encoded, icd9_code_input_encoded, icu_stay_duration, ed_waiting_time, duration_treatment, age
    ]

    if st.button("Predict Length of Stay"):
        predicted_los = predict_los(features)
        normalized_los = normalize_los(predicted_los)
        
        normalized_los_rounded = round(normalized_los)
        st.subheader(f"length of stay: {normalized_los_rounded} jours")

# Fonction pour prédire la durée du séjour (LOS)
def predict_los(features):
    features_scaled = scaler.transform([features])
    predicted_los = model.predict(features_scaled)
    return predicted_los[0]

def main():
    # page = navigation()
    # if page == "Home Page":
    #     home_page()
    # else:
        prediction_page()

if __name__ == "__main__":
    main()
