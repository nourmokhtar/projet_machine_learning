import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

# Configuration de la page
st.set_page_config(page_title="Prédiction du Score de Burnout", layout="wide")

# Charger le modèle
model_path = r"burnout_model.pkl"
model = joblib.load(model_path)

# Fonction de prétraitement des données utilisateur
def preprocess_input(data):
    ordinal_encoders = {
        'Years of service ?': OrdinalEncoder(categories=[["<5 years", "5 to 10 years", "10 -15 years", ">15 years"]])
    }

    label_encoders = {
        'Gender': LabelEncoder(),
        'Previously worked in a critical care unit ? ': LabelEncoder(),
        'Was mental health support available?': LabelEncoder(),
    }

    for col, encoder in ordinal_encoders.items():
        data[col] = encoder.fit_transform(data[[col]])

    for col, encoder in label_encoders.items():
        data[col] = encoder.fit_transform(data[col])

    return data
def prediction_page():
    # Colonnes nécessaires
    required_columns = [
        'Gender', 'Years of service ?', 'Previously worked in a critical care unit ? ',
        'Was mental health support available?', 'Ps1', 'Ps2', 'Ps3', 'Ps4', 'Ps5',
        'Ws1', 'Ws2', 'Ws3', 'Ws4', 'Ws5', 'Ws6', 'Cs1', 'Cs2', 'Cs3', 'Cs4', 'Cs5',
        'Cs6', 'Cs7', 'Cs8', 'Cs9', 'Cs10', 'Cs11', 'Cs13', 'Cs14', 'Cs15'
    ]

    # Interface utilisateur
    user_input = {}
    user_input['Gender'] = st.selectbox("Genre:", ["Male", "Female", "Other"])
    user_input['Years of service ?'] = st.selectbox("Années de service:", ["<5 years", "5 to 10 years", "10 -15 years", ">15 years"])
    user_input['Previously worked in a critical care unit ? '] = st.selectbox("Avez-vous travaillé en soins critiques ?", ["Yes", "No"])
    user_input['Was mental health support available?'] = st.selectbox("Soutien psychologique disponible ?", ["Yes", "No"])


    ps_scores = [1,2,3,4,5]

    # Questions Ps
    ps_questions = [
        "How often are you physically exhausted?",
        "How often are you emotionally exhausted?",
        "How often do you think: 'I can’t take it anymore?'",
        "How often do you feel weak and susceptible to illness?",
        "How often do you feel worn out (extremely tired)?"
    ]

    for i, question in enumerate(ps_questions, start=1):
        response = st.selectbox(question, ["Never", "Rarely", "Sometimes", "Often", "Always"], key=f"ps{i}")
        user_input[f'Ps{i}'] = ps_scores[["Never", "Rarely", "Sometimes", "Often", "Always"].index(response)]

    # Questions Ws
    ws_questions = [
        "Are you exhausted in the morning at the thought of another day at work?",
        "Do you feel that every working hour is tiring for you?",
        "Do you have enough energy for family and friends during leisure time?",
        "Do you feel that your work is emotionally exhausting?",
        "Does your work frustrate you?",
        "Do you feel burnt out (complete physical or mental exhaustion) because of your work?"
    ]

    for i, question in enumerate(ws_questions, start=1):
        response = st.selectbox(question, ["Strongly Disagree", "Disagree", "Undecided", "Agree", "Strongly Agree"], key=f"ws{i}")
        user_input[f'Ws{i}'] = ps_scores[["Strongly Disagree", "Disagree", "Undecided", "Agree", "Strongly Agree"].index(response)]

    # Questions Cs
    cs_questions = [
        "Do you feel it is hard to work in the current scenario?",
        "Does it drain more of your energy to work during the current scenario?",
        "Do you find it fruitful while performing your work during the current scenario?",
        "Do you feel that you are giving more than what you get back while working in the current scenario?",
        "Do you hesitate to work during this current scenario?",
        "Do you feel depressed because of the current scenario?",
        "Do you feel that your patience is tested while working in the current scenario?",
        "Do you feel lockdown due to the current scenario has added stress on you?",
        "Do you feel a lack of support from your organization during the current scenario?",
        "Do you feel more anxious or worried while working in the current scenario?",
        "Do you feel less motivated to complete your tasks during the current scenario?",
        "Do you feel isolated while working in the current scenario?",
        "Do you feel your workload has increased due to the current scenario?",
        "Do you feel uncertain about your future career due to the current scenario?",
        "Do you feel you are being supported by colleagues during the current scenario?"
    ]

    for i, question in enumerate(cs_questions, start=1):
        response = st.selectbox(question, ["Strongly Disagree", "Disagree", "Undecided", "Agree", "Strongly Agree"], key=f"cs{i}")
        user_input[f'Cs{i}'] = ps_scores[["Strongly Disagree", "Disagree", "Undecided", "Agree", "Strongly Agree"].index(response)]

    # Préparation des données
    features_df = pd.DataFrame([user_input], columns=required_columns)
    processed_data = preprocess_input(features_df)

    # Calcul d'une seule caractéristique
    processed_data['single_feature'] = processed_data.mean(axis=1)

    # Adapter les données au modèle
    features_array = processed_data[['single_feature']].values

    # Prédiction
    if st.button("Prédire"):
        try:
            result = model.predict(features_array)
            
            # Obtenir le score de base
            score_base = result[0]

            # Ajouter 30 si le score est entre 0 et 10, sinon ajouter 40
            if 0 <= score_base <= 10:
                score_with_offset = score_base + 30
            else:
                score_with_offset = score_base + 40

            # Normaliser le score pour qu'il reste entre 0 et 100
            score_normalized = min(max(score_with_offset, 0), 100)

            # Afficher le score final
            st.success(f"Votre score de burnout prédit est : {float(score_normalized):.2f}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")

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

