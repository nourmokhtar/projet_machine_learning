import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
import joblib


category = {
    "0xygen": 1, "300": 2, "ADVANCED LIFE SUPPORT DISPOSABLES": 3, "AID AT SCENE": 4, "ALS": 5, "ALS - Disposables": 6, "ALS Disp": 7, 
    "ALS Disp Supplies": 8, "ALS Disp.": 9, "ALS Disposab": 10, "ALS Disposable": 11, "ALS Disposable Supplies": 12, 
    "ALS Disposable Supplies : Endotracheal Intubation": 13, "ALS Disposable Supplies: Endotracheal intubation": 14, 
    "ALS Disposable supplies: Endotracheal Intubation": 15, "ALS Disposable supplies: Endotracheal intubation": 16, 
    "ALS Disposable supplies: IV Drug Therapy": 17, "ALS Disposables": 18, "ALS Disposoal": 19, 
    "ALS ROUTINE DISPOSABLE SUPPLIES": 20, "ALS Routine Disposable Supplies": 21, "ALS SPECIALIZED SERVICE DISPOSABLE SUPPLIES": 22, 
    "ALS SPECIALIZED SERVICE DISPOSABLE SUPPLIES, DEFIBRILLATION": 23, "ALS Special Disposable Supplies :IV Drug Therapy": 24, 
    "ALS Special Disposable Supplies: IV Drug Therapy": 25, "ALS Special Disposable supplies: IV Drug Therapy": 26, 
    "ALS Specialized Service Disposable Supplies": 27, "ALS Specialized Service Disposable Supplies, Defibrillation": 28, 
    "ALS Supplies": 29, "ALS and BLS defibrillation disposables": 30, "ALS disposables": 31, "ALS disposables A0398": 32, 
    "ALS routine disposable supplies": 33, "ALS-E": 34, "ALS-NON": 35, "ALSDISP": 36, "AO2": 37, "AO382": 38, "AO398": 39, 
    "AO422": 40, "Adanced Life Support Disposables": 41, "Adavance life support disposables": 42, "Advance Life Support Disposables": 43, 
    "Advance Life Support disposables": 44, "Advance life support disposable": 45, "Advance life support disposables": 46, 
    "Advanced Life Support Disposable": 47, "Advanced Life Support Disposables": 48, "Advanced Life Support disposables": 49, 
    "Advanced life support disposables": 50, "Aid Call": 51, "Aid Only - No Transport": 52, "Air Medical Assist": 53, 
    "Ambulance Response and Treatment, No Transport": 54, "Ambulance Response, Treatment, No Transport": 55, 
    "Ambulance Stand By, Per Hour": 56, "Ambulance Waiting Time": 57, "Ambulance Waiting time": 58, 
    "Ambulance response and treatment, no transport": 59, "BASIC LIFE SUPPORT DISPOSABLES": 60, "BLS": 61, "BLS Disp": 62, 
    "BLS Disp.": 63, "BLS Disposab": 64, "BLS Disposable": 65, "BLS Disposable Supplies": 66, "BLS Disposables": 67, "BLS Disposal": 68, 
    "BLS ROUTINE DISPOSABLE SUPPLIES": 69, "BLS Routine Disposable Supplies": 70, "BLS Supplies": 71, "BLS disp": 72, 
    "BLS disposable A0382": 73, "BLS disposables": 74, "BLS routine disposable supplies": 75, "BLS supplies": 76, "BLS-E": 77, 
    "BLS-Non": 78, "BLSDisp": 79, "Basic Life Support Disposable": 80, "Basic Life Support Disposables": 81, "Basic Life Support disposables": 82, 
    "Basic life support disposables": 83, "Basic life support dispsables": 84, "Blood Transfusion Services": 85, "CPT": 86, 
    "Declines": 87, "Defibrillation ALS Disposables": 88, "Diff. situation": 89, "Dispatch Fee": 90, 
    "Dispoable supplies: defibrillation ALS and BLS": 91, "Disposable Supplies": 92, 
    "Disposable Supplies: defibrillation ALS and BLS": 93, "Disposable Supplies: defribrillation ALS and BLS": 94, 
    "Disposable supplies: defibrillation ALS": 95, "Disposable supplies: defibrillation ALS and BLS": 96, 
    "ET-3 ALS Treatment in Place": 97, "ET3 Treatment in Place - ALS": 98, "ET3 Treatment in Place - BLS": 99, 
    "EXTRA AMBULANCE ATTENDANT": 100, "Endotracheal Intubation disposables": 101, "Endotracheal intubation disposables": 102, 
    "Esophageal Intubation": 103, "Extra Ambulabce Attendant": 104, "Extra Ambulance Attendant": 105, "Extra Attendant": 106, 
    "Extra Attendant, Ground": 107, "Extra ambulance attendant, ground": 108, "GCPCS": 109, "Glucometer": 110, 
    "HCPC": 111, "HCPCS": 112, "HCPS": 113, "HPCS": 114, "INJECTION, ADENOSINE, 1 MG": 115, 
    "INJECTION, ADENOSINE, 1MG": 116, "INJECTION, AMIODARONE HYDROCHLORIDE, 30 MG": 117, 
    "INJECTION, GLUCAGON HYDROCHLORIDE, PER 1 MG": 118, "INJECTION, GLUCAGON HYDROCHLORIDE, PER 1MG": 119, 
    "INJECTION, METHYLPREDNISOLONE SODIUM SUCCINATE, UP TO 125 MG": 120, "INJECTION, MIDAZOLAM HYDROCHLORIDE, PER 1 MG": 121, 
    "IV Drug Therapy disposable supplies": 122, "IV Drug Therapy disposables": 123, "Injection, Adenosine, 1 MG": 124, 
    "Injection, Amiodarone Hydrochloride, 30 MG": 125, "Injection, Glucagon Hydrochloride, Per 1 MG": 126, 
    "Injection, Methylprednisolone Succinate, up to 125 MG": 127, "Injection, Midazolam Hydrochloride, Per 1 MG": 128, 
    "Intraosseous Needle": 129, "Milage": 130, "Mileage": 131, "Mileage per Loaded Mile": 132, 
    "NEONATAL TRANSPORT, BASE RATE, EMERGENCY TRANSPORT": 133, "NO TRANSPORT WITH TREATMENT": 134, 
    "Neonatal Transport, Base Rate, Emergency Transport": 135, "O 2": 136, "O2": 137, "O2 Used": 138, "O2-": 139, "O2.": 140, 
    "OX2": 141, "OXY2": 142, "OXYGEN": 143, "OXYGEN AND OXYGEN SUPPLIES, LIFE SUSTAINING": 144, "Oxygen": 145, "Oxygen (02)": 146, 
    "Oxygen A0422": 147, "Oxygen Supplies": 148, "Oxygen Use": 149, "Oxygen administration and supplies": 150, 
    "Oxygen and Oxygen Supplies, Life Sustaining": 151, "SCT": 152, "Supplies": 153, "Supply": 154, "TNT": 155, 
    "TNT - TX No Transport": 156, "TNT - Treatment No Transport": 157, "TNT - Treatment No Transport A0998": 158, 
    "TNT- Treatment No Transport": 159, "TNT- Treatment no Transport": 160, "TNT-Treatment No Transport": 161, 
    "TNT/Treatment, No Transport": 162, "TREATMENT, NO TRANSPORT": 163, "TXnoTXP": 164, "Treamtnet, no transport": 165, 
    "Treat No Transport": 166, "Treat, No Transport": 167, "Treat, no Transport": 168, "Treat-no-Transport (TNT)": 169, 
    "Treat/Assist/Aid": 170, "Treat/No Tra": 171, "Treatment No Transport": 172, "Treatment No Transport (TNT)": 173, 
    "Treatment no": 174, "Treatment no Transport": 175, "Treatment no transport": 176, "Treatment, No Transport": 177, 
    "Treatment, No transport": 178, "Treatment, no Transport": 179, "Treatment, no transport": 180, "Treatment-No Transport": 181, 
    "UNCLASSIFIED DRUGS, ADMINISTERED BY INJECTION": 182, "UNLISTED DRUG": 183, "Urgent": 184, "Urinary Catheter": 185, 
    "Urgency": 186
}

# Fonction pour charger et prétraiter les données
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Encodage des colonnes catégorielles
    label_encoder = LabelEncoder()
    df['Code type'] = label_encoder.fit_transform(df['Code type'])
    df['Political subdivision'] = label_encoder.fit_transform(df['Political subdivision'])
    
    # Mise à l'échelle des colonnes numériques
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[['Resident rate', 'Non-Resident Rate', 'Code type', 'Political subdivision']])
    return pd.DataFrame(data_scaled, columns=['Resident rate', 'Non-Resident Rate', 'Code type', 'Political subdivision']), df.columns, scaler, df

# Fonction pour exécuter le clustering
def run_clustering(data_scaled):
    cah = AgglomerativeClustering(n_clusters=2, linkage='ward')  # Par défaut : 2 clusters
    labels = cah.fit_predict(data_scaled)
    return cah, labels

# Fonction pour prédire le cluster d'un nouveau point de données
def predict_cluster(model, scaler, input_data, data_scaled):
    try:
        # Mise à l'échelle de l'entrée utilisateur
        scaled_input = scaler.transform([input_data[:4]])  # Seulement les éléments nécessaires
        
        # Ajout de l'entrée mise à l'échelle au dataset
        augmented_data = np.vstack([data_scaled, scaled_input])
        
        # Ajustement du modèle sur le dataset augmenté
        labels = model.fit_predict(augmented_data)
        
        # Retourne le cluster du dernier point (entrée utilisateur)
        return labels[-1]
    except Exception as e:
        st.error(f"Erreur dans la prédiction : {e}")
        return None
def main():
# Streamlit App
    st.title("The prediction of clusters for political subdivisions.")

    # Charger les données depuis un fichier local
    file_path = "Emergency_Services_Billing_RatesCode_Rates.csv"
    data_scaled, feature_names, scaler, original_df = load_and_preprocess_data(file_path)

    # Clustering
    model, labels = run_clustering(data_scaled)
    original_df['Cluster'] = labels

    # # Afficher les données clusterisées
    # st.subheader("Données clusterisées")
    # st.write(original_df)

    # Formulaire interactif pour prédire un cluster
    with st.form("prediction_form"):
        # 1. Statut : Résident ou Non-Résident
        status = st.selectbox("status", options=["Resident", "Non-Resident"], index=0)

        # 2. Taux en fonction du statut
        if status == "Resident":
            resident_rate = st.number_input("Taux des résidents (Resident rate)", min_value=0.0, step=0.1)
            non_resident_rate = 0.0  # Non applicable pour un résident
        else:
            non_resident_rate = st.number_input("Taux des non-résidents (Non-Resident Rate)", min_value=0.0, step=0.1)
            resident_rate = 0.0  # Non applicable pour un non-résident

        # 3. Autres champs d'entrée
        code_type = st.text_input("code type")
        # Dans la section où vous traitez 'code_type'
        if code_type:
        # Chercher si le code correspond à une catégorie
        # Le code peut être un nombre, donc convertissez-le en entier si nécessaire
            try:
                num_code = int(code_type)  # Essayer de convertir le code en entier
                if num_code in category.values():
                    # Trouver la clé correspondant à la valeur
                    for key, value in category.items():
                        if value == num_code:
                            st.write(f"The code {num_code} corresponds to the category: {key}")
                            break
                else:
                    st.write(f"The code {num_code} is not valid or not in the database.")
            except ValueError:
                    st.write("Please enter a valid code number.")
        political_subdivision = st.text_input("political subdivision")

        # Soumettre le formulaire
        submitted = st.form_submit_button("predict the cluster")

        if submitted:
            try:
                # Préparer les données d'entrée pour la prédiction
                input_data = [resident_rate, non_resident_rate, code_type, political_subdivision]
                cluster = predict_cluster(model, scaler, input_data, data_scaled)
                if cluster is not None:
                    st.success(f"The individual belongs to cluster: {cluster}")

                    # Affichage des informations supplémentaires sur le cluster
                    if cluster == 0:
                        st.write("**Cluster 0**: Groups political subdivisions that apply higher rates for specialized services.")
                    elif cluster == 1:
                        st.write("**Cluster 1**: Groups political subdivisions that apply lower rates for basic services.")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
if __name__ == "__main__":
    main()