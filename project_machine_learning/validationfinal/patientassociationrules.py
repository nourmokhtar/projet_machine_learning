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

# Fonction pour transformer les colonnes numériques en catégories qualitatives
def transform_to_categories(df):
    df['bmi_category'] = pd.cut(df['bmi'], 
                                bins=[0, 18.5, 24.9, 29.9, float('inf')], 
                                labels=["Underweight", "Normal", "Overweight", "Obese"])
    df['age_group'] = pd.cut(df['age'], 
                             bins=[0, 18, 45, 65, float('inf')], 
                             labels=["Child", "Young Adult", "Adult", "Senior"])
    df['blood_pressure_level'] = pd.cut(df['blood_pressure'], 
                                        bins=[0, 120, 140, float('inf')], 
                                        labels=["Normal", "Elevated", "High"])
    df['cholesterol_level'] = pd.cut(df['cholesterol'], 
                                     bins=[0, 200, 240, float('inf')], 
                                     labels=["Normal", "Borderline High", "High"])
    
    # Ajouter une colonne pour indiquer le diabète
    def determine_diabetes(row):
        if row['plasma_glucose'] >= 126:
            return "Diabetic"
        elif 100 <= row['plasma_glucose'] < 126:
            return "Prediabetic"
        else:
            return "Non-Diabetic"
    
    df['diabetes_status'] = df.apply(determine_diabetes, axis=1)
    return df

# Fonction pour transformer les données en transactions
def transform_to_transactions(df):
    qualitative_columns = ['gender','age_group', 'bmi_category','diabetes_status', 'blood_pressure_level', 
                           'cholesterol_level','heart_disease', 'smoking_status']
    transactions = df[qualitative_columns].applymap(str).values.tolist()
    return transactions

# Fonction pour appliquer l'algorithme Apriori
def generate_apriori_rules(transactions):
    # Étape 1 : Conversion des transactions en format One-Hot Encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Appliquer l'algorithme Apriori (pour générer les itemsets fréquents)
    frequent_itemsets = apriori(df, min_support=0.35, use_colnames=True)

    # Appliquer l'algorithme de règles d'association pour générer des règles
    rules = association_rules(frequent_itemsets, num_itemsets=len(frequent_itemsets), metric="confidence", min_threshold=0.5)
    
    # Retourner les règles
    return rules

# Initialisation des encodeurs
label_encoder_diagnosis = LabelEncoder()
label_encoder_description = LabelEncoder()
label_encoder_discharge_location = LabelEncoder()

        
def association_page():
        # Interface utilisateur Streamlit
    st.title("Application de Règles d'Association - Apriori")

    # Demander à l'utilisateur de télécharger un fichier CSV
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type="csv")

    if uploaded_file is not None:
        # Charger le fichier CSV téléchargé
        df = pd.read_csv(uploaded_file)
        st.write(f"Fichier téléchargé : {uploaded_file.name}")
        
        # Nettoyage des données et transformation en catégories qualitatives
        df_cleaning = df.dropna()  # Dropping rows with missing values (optional)
        transformed_data = transform_to_categories(df_cleaning)

        # Transformer les données en transactions
        transactions = transform_to_transactions(transformed_data)
        


        # Appliquer l'algorithme Apriori et générer les règles
        rules = generate_apriori_rules(transactions)
        
        # Afficher les règles générées
        if not rules.empty:
            st.header("Règles d'Association")
            # Option pour filtrer les règles en fonction du seuil de confiance
            confidence_threshold = st.slider("Sélectionner le seuil de confiance", min_value=0.0, max_value=1.0, step=0.01, value=0.6)
            filtered_rules = rules[rules['confidence'] >= confidence_threshold]
            st.write(f"Règles filtrées avec confiance >= {confidence_threshold}")
            st.write(filtered_rules)
        else:
            st.write("Aucune règle générée. Vérifiez le fichier CSV.")


# Fonction principale
def main():
        association_page() 


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