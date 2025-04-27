import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder , StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules


apscaler = StandardScaler()




def apriori_delay():
    st.title("Association Rules of Delay in Few Events.")

    # Téléchargement des fichiers CSV
    uploaded_file_delay  = st.file_uploader("Télécharger le fichier d'exemple d'evenement :", type="csv")
    # uploaded_file_prescriptions = st.file_uploader("Télécharger le fichier des prescriptions (PRESCRIPTIONS1.csv)", type="csv")
    
    if uploaded_file_delay:
        # Lire les fichiers CSV
        delay_df = pd.read_csv(uploaded_file_delay)

        
        # # Assurez-vous que 'hadm_id' soit de type str
        # diagnoses['hadm_id'] = diagnoses['hadm_id'].astype(str)
        # # prescriptions['hadm_id'] = prescriptions['hadm_id'].astype(str)
        
        # # Fusionner les DataFrames sur 'hadm_id'
        # merged_data = pd.concat([diagnoses[['hadm_id', 'long_title']], prescriptions[['hadm_id', 'drug']]], axis=1)
        # merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]  # Supprimer les doublons de colonnes
        # merged_data = merged_data.dropna(subset=['hadm_id'])  # Enlever les lignes avec 'hadm_id' manquant
        
        # # Appliquer l'algorithme Apriori
        # min_support = st.slider("Seuil de support minimum", 0.0, 1.0, 0.04)
        
        if st.button("Générer des Règles d'Association"):
            delay_df=delay_df.drop(delay_df[~delay_df.linksto.isin(['chartevents','datetimeevents','inputevents_cv','outputevents','inputevents_mv','procedureevents_mv','microbiologyevents'])].index )
            delay_df=delay_df.drop_duplicates()

            d=delay_df['date']
            delay_df['date'] = pd.to_datetime(delay_df['date'])
            delay_df['year']=delay_df['date'].dt.year
            delay_df['month']=delay_df['date'].dt.month
            delay_df['day']=delay_df['date'].dt.day
            delay_df['timestamp'] = delay_df['date'].apply(lambda x: x.timestamp())
            delay_df = delay_df.drop('date', axis=1)

            ordinal = OrdinalEncoder(categories=[['chartevents','datetimeevents','inputevents_cv','inputevents_mv','microbiologyevents','procedureevents_mv','outputevents']])
            delay_df.linksto = ordinal.fit_transform(delay_df.linksto.values.reshape(-1, 1))

            ordinal = OrdinalEncoder(categories=[['chartevent','datetimeevent','inputevent_cv1','inputevent_mv1','microbiologyevent','procedureevent','outputevent']])
            delay_df.eventtype = ordinal.fit_transform(delay_df.eventtype.values.reshape(-1, 1))

            
            le = LabelEncoder()
            delay_df['label'] = le.fit_transform(delay_df['label'])

            # prescriptions = pd.read_csv(uploaded_file_prescriptions)
            AR_data = apscaler.fit_transform(delay_df)
            AR_df = pd.DataFrame(AR_data, columns=delay_df.columns)

            #from floats to "classes"
            AR_df['value_binned'] = pd.cut(AR_df['value'], bins=3, labels=['low', 'medium', 'high'])
            AR_df['label_binned'] = pd.cut(AR_df['label'], bins=3, labels=['low', 'medium', 'high'])
            AR_df['timestamp_binned'] = pd.cut(AR_df['timestamp'], bins=3, labels=['low', 'medium', 'high'])
            AR_df['first_wardid_binned'] = pd.cut(AR_df['first_wardid'], bins=3, labels=['low', 'medium', 'high'])
            AR_df['year_binned'] = pd.cut(AR_df['year'], bins=3, labels=['low', 'medium', 'high'])
            AR_df['month_binned'] = pd.cut(AR_df['month'], bins=3, labels=['low', 'medium', 'high'])
            AR_df['day_binned'] = pd.cut(AR_df['day'], bins=3, labels=['low', 'medium', 'high'])

            #remove original
            binned_data = AR_df[['value_binned', 'label_binned', 'timestamp_binned','first_wardid_binned','year_binned','month_binned','day_binned']]

            # Use pandas get_dummies for one-hot encoding
            encoded_df = pd.get_dummies(binned_data)
            delay_frequent_items = apriori(encoded_df, min_support=0.4, use_colnames=True)
            delay_rules = association_rules(delay_frequent_items, metric="confidence", min_threshold=1,num_itemsets=7)
            
        
            # rules = apply_apriori(merged_data, min_support)
            
            # Afficher les règles générées
            st.subheader("Règles d'Association")
            
            if not delay_rules.empty:
                for index, row in delay_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('support',ascending=False).head(5).iterrows():
                    antecedents = list(row['antecedents'])
                    consequents = list(row['consequents'])
                    msg =f"If {', '.join(antecedents)} then {', '.join(consequents)}"
                    msg =msg.replace( "value_binned_low","delay is low ")
                    msg =msg.replace( "value_binned_medium","delay is medium")
                    msg =msg.replace( "value_binned_high","delay is high")

                    msg =msg.replace( "timestamp_binned_low","it is early in the day")
                    msg =msg.replace( "timestamp_binned_medium","it is midday")
                    msg =msg.replace( "timestamp_binned_high","it is late/end of the day")

                    msg =msg.replace( "year_binned_low","year is 1950-1980")
                    msg =msg.replace( "year_binned_medium","year is 1985-2030")
                    msg =msg.replace( "year_binned_high","year is 2030-2050")

                    msg =msg.replace( "month_binned_low","it is winter season")
                    msg =msg.replace( "month_binned_medium","it is summer season")
                    msg =msg.replace( "month_binned_high","it is end of year")

                    
                    msg =msg.replace( "first_wardid_binned_low","ward is close/on a low floor")
                    msg =msg.replace( "first_wardid_binned_medium","ward is in the middle floors")
                    msg =msg.replace( "first_wardid_binned_high","ward is far/on a high floor")


                    
                    msg =msg.replace( "day_binned_low","it is first/start of the month")
                    msg =msg.replace( "day_binned_medium","it is middle of the month")
                    msg =msg.replace( "day_binned_high","it is end of the month")
                    st.write(msg)
            else:
                st.write("Aucune règle trouvée avec ce seuil de support.")
  





# Fonction principale
def main():
    apriori_delay()
    

# Lancer l'application
if __name__ == "__main__":
    # Style personnalisé
    main()