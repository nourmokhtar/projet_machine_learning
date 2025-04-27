import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

# Charger le modèle, le scaler et le PCA
model = joblib.load('random_forest_model.pkl')

# Initialisation des encodeurs
label_encoder_diagnosis = LabelEncoder()
label_encoder_description = LabelEncoder()
label_encoder_discharge_location = LabelEncoder()

# Exemple d'encodage pour diagnosis, description, discharge_location
label_encoder_discharge_location.fit(['HOME HEALTH CARE', 'DEAD/EXPIRED', 'SNF',
       'REHAB/DISTINCT PART HOSP', 'HOME', 'HOSPICE-HOME',
       'DISCH-TRAN TO PSYCH HOSP', 'LONG TERM CARE HOSPITAL', 'ICF'])  # Ajustez selon vos valeurs
label_encoder_description.fit(['SEPTICEMIA AGE >17', 'Shoulder, Upper Arm & Forearm Procedures',
       'MAJOR JOINT & LIMB REATTACHMENT PROCEDURES OF UPPER EXTREMITY',
       'Intracranial Hemorrhage',
       'INTRACRANIAL HEMORRHAGE OR CEREBRAL INFARCTION',
       'Cardiac Valve Procedures w/o Cardiac Catheterization',
       'CARDIAC VALVE & OTHER MAJOR CARDIOTHORACIC PROC WITHOUT CARDIAC CATHETER',
       'Infectious & Parasitic Diseases Including HIV W O.R. Procedure',
       'INFECTIOUS & PARASITIC DISEASES W OR PROCEDURE',
       'Other Endocrine Disorders',
       'ENDOCRINE DISORDERS WITH COMPLICATIONS, COMORBIDITIES',
       'Malfunction, Reaction & Comp of Orthopedic Device or Procedure',
       'AFTERCARE, MUSCULOSKELETAL SYSTEM & CONNECTIVE TISSUE',
       'RESPIRATORY SYSTEM DIAGNOSIS WITH VENTILATOR SUPPORT',
       'CIRRHOSIS & ALCOHOLIC HEPATITIS', 'Other Disorders Of The Liver',
       'OTHER MULTIPLE SIGNIFICANT TRAUMA',
       'Multiple Significant Trauma W/O O.R. Procedure',
       'Major Stomach, Esophageal & Duodenal Procedures',
       'STOMACH, ESOPHAGEAL & DUODENAL PROC AGE >17 W CC W/O MAJOR GI DX',
       'EXTENSIVE OPERATING ROOM PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS',
       'Connective Tissue Disorders',
       'CONNECTIVE TISSUE DISORDERS WITH COMPLICATIONS, COMORBIDITIES',
       'CIRCULATORY DISORDERS WITH ACUTE MYOCARDIAL INFARCTION & MAJOR COMPLICATION, DISCHARGED ALIVE',
       'POISONING & TOXIC EFFECTS OF DRUGS AGE >17 WITH COMPLICATIONS, COMORBIDITIES',
       'Poisoning Of Medicinal Agents',])  # Ajustez selon vos valeurs
label_encoder_diagnosis.fit(['SEPSIS', 'HUMERAL FRACTURE', 'STROKE/TIA',
       ' MITRAL REGURGITATION;CORONARY ARTERY DISEASE\\CORONARY ARTERY BYPASS GRAFT WITH MVR  ? MITRAL VALVE REPLACEMENT /SDA',
       'SYNCOPE;TELEMETRY', 'RENAL FAILIURE-SYNCOPE-HYPERKALEMIA',
       'FAILURE TO THRIVE', 'RESPIRATORY DISTRESS', 'FEVER',
       'VARICEAL BLEED', 'LOWER GI BLEED', 'SUBDURAL HEMATOMA/S/P FALL',
       'ESOPHAGEAL CANCER/SDA', 'GASTROINTESTINAL BLEED', 'HYPOTENSION',
       'CONGESTIVE HEART FAILURE', 'OVERDOSE',
       'CRITICAL AORTIC STENOSIS/HYPOTENSION', 'HYPOTENSION;TELEMETRY',
       'SEPSIS;TELEMETRY',
       'STATUS POST MOTOR VEHICLE ACCIDENT WITH INJURIES',
       'TACHYPNEA;TELEMETRY', 'LIVER FAILURE', 'LEFT HIP FRACTURE',
       'S/P MOTOR VEHICLE ACCIDENT', 'SHORTNESS OF BREATH', 'PNEUMONIA',
       'ACUTE CHOLANGITIS', 'FEVER;URINARY TRACT INFECTION',
       'SYNCOPE;TELEMETRY;INTRACRANIAL HEMORRHAGE', 'LEFT HIP OA/SDA',
       'MEDIASTINAL ADENOPATHY', 'FACIAL NUMBNESS',
       'AROMEGLEY;BURKITTS LYMPHOMA', 'STEMI;'])  # Ajustez selon vos valeurs

# Fonction pour encoder les colonnes
def encode_diagnosis(diagnosis):
    return label_encoder_diagnosis.transform([diagnosis])[0]

def encode_description(description):
    return label_encoder_description.transform([description])[0]

def encode_discharge_location(location):
    return label_encoder_discharge_location.transform([location])[0]

# Fonction pour l'encodage One-Hot de l'assurance
def encode_insurance(insurance):
    insurance_mapping = {
        'Medicare': [1, 0, 0, 0],
        'Private': [0, 1, 0, 0],
        'Government': [0, 0, 1, 0],
        'Medicaid': [0, 0, 0, 1],
    }
    return insurance_mapping[insurance]

# Fonction pour encoder antibio_flag et hospital_expire_flag
def encode_flags(antibio_flag, hospital_expire_flag):
    return 1 if antibio_flag else 0, 1 if hospital_expire_flag == "Yes" else 0

# Fonction pour la prédiction
def predict_readmission(features):
    prediction = model.predict([features])
    return "Réadmis" if prediction[0] == 1 else "Non réadmis"

  

# Page de prédiction
def prediction_page():
    st.title("Prédiction de Réadmission Hospitalière")

    # Entrées utilisateur pour chaque feature
    subject_id = st.number_input("ID du Patient (Subject ID)", min_value=0, step=1)
    hadm_id = st.number_input("ID d'Hospitalisation (HADM ID)", min_value=0, step=1)
    admission_type = st.selectbox("Type d'Admission", options=[0, 1], format_func=lambda x: "ÉLECTIVE" if x == 0 else "URGENCE")

    # Sélection du type d'assurance
    insurance = st.selectbox("Type d'Assurance", options=['Medicare', 'Private', 'Government', 'Medicaid', 'Other'])

    # Sélection du diagnostic, description, et lieu de sortie
    diagnosis = st.selectbox("Diagnostic", options=label_encoder_diagnosis.classes_)
    description = st.selectbox("Description", options=label_encoder_description.classes_)
    discharge_location = st.selectbox("Lieu de sortie", options=label_encoder_discharge_location.classes_)

    # Sélection de la sévérité et de la mortalité du DRG
    drg_severity = st.slider("Sévérité du DRG", min_value=1, max_value=4, step=1)
    drg_mortality = st.slider("Mortalité du DRG", min_value=1, max_value=4, step=1)

    # Sélection de l'antibiotique et de la sortie à l'hôpital
    antibio_flag = st.selectbox("Antibiotique administré", options=[True, False])
    hospital_expire_flag = st.selectbox("Décédé à l'hôpital ?", options=["Yes", "No"])

    # Sélection du calendrier pour la date de sortie
    disch_date = st.date_input("Date de sortie")

    # Sélection de l'heure et de la minute via des selectbox
    hours = list(range(24))  # De 0 à 23 heures
    minutes = list(range(0, 60, 5))  # Par intervalles de 5 minutes
    disch_hour = st.selectbox("Heure de sortie", options=hours)
    disch_minute = st.selectbox("Minute de sortie", options=minutes)

    # Combiner la date et l'heure
    disch_datetime = pd.to_datetime(f"{disch_date} {disch_hour}:{disch_minute}")
    # Extraire jour, mois, année, heure et minute de la date et heure
    disch_day = disch_date.day
    disch_month = disch_date.month
    disch_year = disch_date.year

    # Encodage des colonnes
    encoded_diagnosis = encode_diagnosis(diagnosis)
    encoded_description = encode_description(description)
    encoded_discharge_location = encode_discharge_location(discharge_location)

    # Encodage One-Hot de l'assurance
    encoded_insurance = encode_insurance(insurance)

    # Encodage des flags
    encoded_antibio_flag, encoded_hospital_expire_flag = encode_flags(antibio_flag, hospital_expire_flag)

    # Préparer les caractéristiques pour la prédiction
    features = [
        subject_id, hadm_id, admission_type, *encoded_insurance, encoded_diagnosis,
        encoded_description, drg_severity, drg_mortality, encoded_discharge_location, disch_day,
        disch_month, disch_year, disch_hour, disch_minute, encoded_antibio_flag, encoded_hospital_expire_flag
    ]

    # Bouton de prédiction
    if st.button("Prédire"):
        result = predict_readmission(features)
        st.subheader("Résultat de la Prédiction :")
        st.success(result)


# Fonction principale
def main():
    prediction_page()



# Lancer l'application
if __name__ == "__main__":
    # Style personnalisé
    st.markdown("""
        <style>
          
            .stButton>button {
                background-color: #ed2f2f;
                color: white;
                font-size: 18px;
                border-radius: 8px;
                padding: 10px 20px;
            }
            .stSidebar {
                background-color: #333;
                color: white;
            }
            .stTitle {
                font-size: 36px;
                font-weight: bold;
                color: #00BFAE;
            }
            .stSubheader {
                color: #FFEB3B;
            }
            .stTextInput input {
                background-color: #333;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    main()