import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Fonction pour appliquer l'algorithme Apriori
def apply_apriori(data, min_support=0.04):
    # Créer des transactions en combinant les diagnostics et les médicaments
    transactions = data.groupby("hadm_id").apply(
    lambda x: set(x["long_title"].dropna().astype(str)) | set(x["drug"].dropna().astype(str))
    ).reset_index(name="Items")

    # Filtrer les items rares
    item_counts = transactions['Items'].explode().value_counts()
    filtered_items = item_counts[item_counts > 10].index  # Ne garder que les items fréquents
    transactions['Items'] = transactions['Items'].apply(
        lambda x: [item for item in x if item in filtered_items]
    )
    transactions = transactions[transactions['Items'].str.len() > 0]  # Supprimer les transactions vides
    
    # Encodage des transactions
    transactions_list = transactions['Items'].tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions_list).transform(transactions_list)
    encoded_transactions = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Appliquer Apriori pour générer des items fréquents
    frequent_itemsets = apriori(encoded_transactions, min_support=min_support, use_colnames=True)
    
    # Générer des règles d'association
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=2)
    
    return rules

# Fonction principale de l'application Streamlit
def main():
    st.title("Association de Médicaments et Diagnostics en Soins Intensifs")

    # Téléchargement des fichiers CSV
    uploaded_file_diagnoses = st.file_uploader("Télécharger le fichier des diagnostics (DIAGNOSES_ICD.csv)", type="csv")
    uploaded_file_prescriptions = st.file_uploader("Télécharger le fichier des prescriptions (PRESCRIPTIONS1.csv)", type="csv")
    
    if uploaded_file_diagnoses and uploaded_file_prescriptions:
        # Lire les fichiers CSV
        diagnoses = pd.read_csv(uploaded_file_diagnoses)
        prescriptions = pd.read_csv(uploaded_file_prescriptions)
        
        # Assurez-vous que 'hadm_id' soit de type str
        diagnoses['hadm_id'] = diagnoses['hadm_id'].astype(str)
        prescriptions['hadm_id'] = prescriptions['hadm_id'].astype(str)
        
        # Fusionner les DataFrames sur 'hadm_id'
        merged_data = pd.concat([diagnoses[['hadm_id', 'long_title']], prescriptions[['hadm_id', 'drug']]], axis=1)
        merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]  # Supprimer les doublons de colonnes
        merged_data = merged_data.dropna(subset=['hadm_id'])  # Enlever les lignes avec 'hadm_id' manquant
        
        # Appliquer l'algorithme Apriori
        min_support = st.slider("Seuil de support minimum", 0.0, 1.0, 0.04)
        
        if st.button("Générer des Règles d'Association"):
            rules = apply_apriori(merged_data, min_support)
            
            # Afficher les règles générées
            st.subheader("Règles d'Association")
            
            if not rules.empty:
                for index, row in rules.iterrows():
                    antecedents = list(row['antecedents'])
                    consequents = list(row['consequents'])
                    st.write(f"Si {', '.join(antecedents)} alors {', '.join(consequents)}")
            else:
                st.write("Aucune règle trouvée avec ce seuil de support.")
                
# Lancer l'application Streamlit
if __name__ == "__main__":
    main()
