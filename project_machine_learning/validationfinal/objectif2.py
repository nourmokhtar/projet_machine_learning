import streamlit as st
import joblib
import numpy as np  
import pandas as pd
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
model = joblib.load('decision_tree_model.pkl')  # Assurez-vous que votre modèle est un modèle de prédiction de admission_type_URGENT
scaler = joblib.load('minmax_scaler.pkl')  # Assurez-vous que votre modèle est un modèle de prédiction de admission_type_URGENT

# Initialisation des encodeurs LabelEncoder pour les colonnes catégorielles
label_encoder_admission_type = LabelEncoder()
label_encoder_insurance = LabelEncoder()
label_encoder_first_careunit = LabelEncoder()
label_encoder_last_careunit = LabelEncoder()
label_encoder_expire_flag = LabelEncoder()
label_encoder_hospital_expire_flag = LabelEncoder()
label_encoder_discharge_location = LabelEncoder()


# Ajustement des encodeurs avec les valeurs possibles
label_encoder_admission_type.fit(['EMERGENCY', 'ELECTIVE'])
label_encoder_insurance.fit(['Medicare', 'Private', 'Medicaid', 'Government'])
label_encoder_first_careunit.fit(['MICU', 'SICU', 'TSICU', 'CSRU', 'CCU'])
label_encoder_last_careunit.fit(['MICU', 'SICU', 'TSICU', 'CSRU', 'CCU'])
label_encoder_expire_flag.fit([1, 0])
label_encoder_hospital_expire_flag.fit([1, 0])
label_encoder_discharge_location.fit (['SNF','DEAD/EXPIRED','LONG TERM CARE HOSPITAL','REHAB/DISTINCT PART HOSP',
 'HOME', 'HOME HEALTH CARE','HOME WITH HOME IV PROVIDR','DISCH-TRAN TO PSYCH HOSP'])

# Fonction pour gérer les valeurs inconnues avec LabelEncoder
def encode_with_unknown(encoder, data):
    try:
        return encoder.transform(data)
    except ValueError as e:
        print(f"Erreur d'encodage pour '{data}': {e}")
        # Retourner une valeur par défaut pour les valeurs inconnues
        return np.array([-1] * len(data))  # -1 comme valeur par défaut
    
# # Définir la navigation
# def navigation():
#     st.sidebar.title("Menu")
#     page = st.sidebar.radio("", ["Home Page", "Admission Type Prediction"])
#     return page

# # Page d'accueil
# def home_page():
#     st.title("Welcome to the Admission Type Prediction App")
#     st.subheader("We predict the patient's admission type based on various features")

def prediction_page():
    st.title("Admission Type Prediction")

    # Entrées utilisateur pour chaque feature avec des clés uniques
    discharge_location = st.selectbox("Discharge Location", options=['SNF', 'DEAD/EXPIRED', 'LONG TERM CARE HOSPITAL', 'REHAB/DISTINCT PART HOSP', 'HOME', 'HOME HEALTH CARE', 'HOME WITH HOME IV PROVIDR', 'DISCH-TRAN TO PSYCH HOSP'], key="discharge_location")
    insurance = st.selectbox("Insurance Type", options=['Medicare', 'Private', 'Medicaid', 'Government'], key="insurance")
    hospital_expire_flag = st.selectbox("Deceased at Hospital?", options=[1, 0], key="hospital_expire_flag")  # Hospital expire flag
    expire_flag = st.selectbox("Deceased?", options=[1, 0], key="expire_flag")  # Expire flag
    first_careunit = st.selectbox("First Care Unit", options=['MICU', 'SICU', 'TSICU', 'CSRU', 'CCU'], key="first_careunit")
    last_careunit = st.selectbox("Last Care Unit", options=['MICU', 'SICU', 'TSICU', 'CSRU', 'CCU'], key="last_careunit")
    age = st.number_input("Patient Age", min_value=0, step=1, key="age")
    los = st.number_input("Length of Stay (LOS) in days", min_value=0, step=1, key="los")
    icu_stay_duration = st.number_input("ICU Stay Duration (in days)", min_value=0, step=1, key="icu_stay_duration")
    ed_waiting_time = st.number_input("Emergency Department Waiting Time (in minutes)", min_value=0, step=1, key="ed_waiting_time")
    
   

  

    # Encoder les variables catégorielles existantes
    discharge_location_encoded = encode_with_unknown(label_encoder_discharge_location, [discharge_location])[0]
    insurance_encoded = encode_with_unknown(label_encoder_insurance, [insurance])[0]
    first_careunit_encoded = encode_with_unknown(label_encoder_first_careunit, [first_careunit])[0]
    last_careunit_encoded = encode_with_unknown(label_encoder_last_careunit, [last_careunit])[0]
    expire_flag_encoded = encode_with_unknown(label_encoder_expire_flag, [expire_flag])[0]
    hospital_expire_flag_encoded = encode_with_unknown(label_encoder_hospital_expire_flag, [hospital_expire_flag])[0]

    # Préparer les features pour la prédiction
    features = [
        discharge_location_encoded, insurance_encoded, hospital_expire_flag_encoded, expire_flag_encoded,
        first_careunit_encoded, last_careunit_encoded, age, los, icu_stay_duration, ed_waiting_time  # Ajouter drug et route
    ]
 
    # Vérifier que vous utilisez 15 caractéristiques ici
    print(len(features))  # Cela devrait afficher 15

    if st.button("Predict Admission Type"):
        # Prédire l'admission_type
        predicted_admission_type = predict_admission_type(features)
        # Afficher la prédiction
        admission_type_predicted = label_encoder_admission_type.inverse_transform([predicted_admission_type])[0]
        st.subheader(f"Predicted Admission Type: {admission_type_predicted}")
        
# Fonction pour prédire admission_type
def predict_admission_type(features):
    # Normaliser les features
    features_scaled = scaler.transform([features])
    predicted_admission_type = model.predict(features_scaled)
    return predicted_admission_type[0]


# Fonction principale
def main():
    # page = navigation()
    # if page == "Home Page":
    #     home_page()
    # else:
    prediction_page()

if __name__ == "__main__":
    main()
