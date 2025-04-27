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
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder




gmm = joblib.load("gmm_model.pkl")  # Charger le modèle GMM
scaler = joblib.load("scalerrania.pkl")  # Charger le scaler
pca = joblib.load("pca.pkl")  # Charger l'objet PCA
# Fonction pour prédire le cluster
def predict_cluster(features):
    # Mise à l'échelle des données
    scaled_features = scaler.transform([features])
    # Réduction de dimension avec PCA
    pca_features = pca.transform(scaled_features)
    # Prédiction du cluster avec le modèle GMM
    cluster = gmm.predict(pca_features)
    return cluster[0]

# Initialisation des encodeurs
label_encoder_diagnosis = LabelEncoder()
label_encoder_description = LabelEncoder()
label_encoder_discharge_location = LabelEncoder()



def clustering_page():
    st.title("Patient clustering")
    st.write("Enter the features to get the corresponding cluster.")
        # User Interface for Data Input
    age = st.number_input("Age", min_value=0, max_value=120, step=1)

    # Gender selection
    gender_mapping = {"Male": 1, "Female": 0}
    gender_text = st.selectbox("Gender", options=["Male", "Female"])
    gender = gender_mapping[gender_text]

    # Chest pain type
    chest_pain_type = st.number_input("Chest Pain Type (1-4)", min_value=1, max_value=4, step=1)

    # Blood pressure
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=300, step=1)

    # Cholesterol
    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=500, step=1)

    # Maximum heart rate
    max_heart_rate = st.number_input("Maximum Heart Rate", min_value=0, max_value=300, step=1)

    # Exercise-induced angina
    exercise_angina_mapping = {"No": 0, "Yes": 1}
    exercise_angina_text = st.selectbox("Exercise-Induced Angina", options=["No", "Yes"])
    exercise_angina = exercise_angina_mapping[exercise_angina_text]

    # Plasma glucose
    plasma_glucose = st.number_input("Plasma Glucose", min_value=0.0, max_value=500.0, step=0.1)

    # Skin thickness
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, step=0.1)

    # Insulin
    insulin = st.number_input("Insulin", min_value=0.0, max_value=500.0, step=0.1)

    # BMI
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=100.0, step=0.1)

    # Diabetes pedigree
    diabetes_pedigree = st.number_input("Diabetes Pedigree", min_value=0.0, max_value=5.0, step=0.01)

    # Hypertension
    hypertension_mapping = {"No": 0, "Yes": 1}
    hypertension_text = st.selectbox("Hypertension", options=["No", "Yes"])
    hypertension = hypertension_mapping[hypertension_text]

    # Heart disease
    heart_disease_mapping = {"No": 0, "Yes": 1}
    heart_disease_text = st.selectbox("Heart Disease", options=["No", "Yes"])
    heart_disease = heart_disease_mapping[heart_disease_text]

    # Smoking status
    smoking_status_mapping = {"Smoker": 1, "Non-Smoker": 0, "Unknown": -1}
    smoking_status_text = st.selectbox("Smoking Status", options=["Smoker", "Non-Smoker", "Unknown"])
    smoking_status = smoking_status_mapping[smoking_status_text]

    # Mapping des interprétations des clusters
    cluster_interpretations = {
        0: "Cluster 0: This group primarily consists of elderly individuals with high cholesterol levels, suffering from heart diseases and diabetes. Additionally, their heart rates are generally higher than average.",
        1: "Cluster 1: This cluster includes individuals of all age groups. Their cholesterol levels are generally normal, but a significant proportion suffers from heart diseases, obesity, and diabetes.",
        2: "Cluster 2: This group mainly consists of young adults with blood pressure often near normal and average cholesterol levels. Cases of obesity, diabetes, and heart diseases are significantly less frequent compared to other clusters."
    }

    # Bouton pour soumettre les données
    if st.button("Predict Cluster"):
        # Rassembler les données en entrée
        input_data = np.array([[age, gender, chest_pain_type, blood_pressure, cholesterol,
                                max_heart_rate, exercise_angina, plasma_glucose,
                                skin_thickness, insulin, bmi, diabetes_pedigree,
                                hypertension, heart_disease, smoking_status]])
        
        # Normaliser les données
        input_scaled = scaler.transform(input_data)
        
        # Réduire les dimensions avec PCA
        input_pca = pca.transform(input_scaled)
        
        # Prédire le cluster avec le modèle GMM
        predicted_cluster = gmm.predict(input_pca)[0]
        
        # Afficher le cluster prédit
        st.write(f"The predicted cluster is: Cluster {predicted_cluster}")
        
        # Afficher l'interprétation correspondante
        st.write(cluster_interpretations[predicted_cluster])



# Fonction principale
def main():
    clustering_page()  


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

