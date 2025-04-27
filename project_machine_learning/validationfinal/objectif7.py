import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

def apply_apriori(data, min_support=0.3):
    transactions = []
    # Ensure we only add rows where there's a 1 for the symptoms
    for _, row in data.iterrows():
        transaction = [column for column in data.columns if row[column] == 1]
        transactions.append(transaction)

    # Apply TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Apply Apriori Algorithm to find frequent itemsets
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    return frequent_itemsets

def load_model():
    try:
        model = joblib.load("frequent_itemsets.pkl")  # Adjust the file path to where your model is stored
        st.write(model)  # Print the entire model to inspect its structure
        
        # If the model is a dictionary or any other structure, check the keys
        if isinstance(model, dict):
            if 'frequent_itemsets' in model and 'rules' in model:
                frequent_itemsets = model['frequent_itemsets']
                rules = model['rules']
                return frequent_itemsets, rules
            else:
                st.error("Expected keys 'frequent_itemsets' and 'rules' not found in the model.")
                return None, None
        else:
            st.error("Model format is not as expected.")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


# Apriori App Function
def main():
    st.title("Symptom Association Finder")

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the dataset
        data = pd.read_csv(uploaded_file)

        # Clean the dataset by dropping unwanted columns
        if 'Contact_Dont-Know' in data.columns:
            data = data.drop(columns=['Contact_Dont-Know', 'Contact_No', 'Contact_Yes', 'Country', 'Gender_Transgender', 'None_Experiencing'])

        # Apply Apriori Algorithm
        min_support = st.slider("Set Symptom Association Threshold", 0.0, 1.0, 0.3)

        if st.button("Find Symptom Pairs"):
            frequent_itemsets = apply_apriori(data, min_support)

            # Display symptom pairs (frequent itemsets) that are found together
            st.subheader("Frequently Occurring Symptom Pairs")

            # Filter the frequent itemsets with more than one symptom
            symptom_pairs = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 2)]

            if symptom_pairs.empty:
                st.write("No symptom pairs found with the given threshold.")
            else:
                for index, row in symptom_pairs.iterrows():
                    symptoms = list(row['itemsets'])
                    st.write(f"Symptoms: {', '.join(symptoms)}")

    # Option to load pre-trained .pkl models (if needed)
    if st.button("Load Pre-trained Model"):
        frequent_itemsets, rules = load_model()

        # Display the pre-trained frequent itemsets and rules
        if frequent_itemsets is not None:
            st.subheader("Pre-trained Symptom Pairs")
            for index, row in frequent_itemsets.iterrows():
                symptoms = list(row['itemsets'])
                st.write(f"Symptoms: {', '.join(symptoms)}")

if __name__ == "__main__":
    main()
