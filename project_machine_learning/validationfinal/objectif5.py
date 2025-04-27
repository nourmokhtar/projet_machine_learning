import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
# Load the models
premium_model_path = r"regression_randomforest.pkl"  # Update with your premium prediction model

premium_model = joblib.load(premium_model_path)
# Prediction function for premium prediction model
def predict_premium(features, model):
    try:
        return model.predict(features)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None
def  main():
    st.title("Health Insurance Premium Prediction")
    
    # Collect user inputs with user-friendly options
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    
    # Convert the selections to 0 or 1 for the model
    blood_pressure = st.selectbox("Blood Pressure Problems", options=["Yes", "No"])
    blood_pressure = 1 if blood_pressure == "Yes" else 0
    
    transplant = st.selectbox("Any Transplants", options=["Yes", "No"])
    transplant = 1 if transplant == "Yes" else 0
    
    chronic_diseases = st.selectbox("Any Chronic Diseases", options=["Yes", "No"])
    chronic_diseases = 1 if chronic_diseases == "Yes" else 0
    
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    
    cancer_in_family = st.selectbox("History of Cancer in Family", options=["Yes", "No"])
    cancer_in_family = 1 if cancer_in_family == "Yes" else 0
    
    surgeries = st.number_input("Number of Major Surgeries", min_value=0, max_value=10, value=0)

    # Prepare input data for prediction
    premium_input_data = np.array([[age, blood_pressure, transplant, chronic_diseases, weight, cancer_in_family, surgeries]])

    # Button to trigger prediction
    if st.button("Predict Health Insurance Premium"):
        premium_result = predict_premium(premium_input_data, premium_model)
        if premium_result is not None:
            st.write(f"Predicted Premium Price: ${premium_result[0]:.2f}")

if __name__ == "__main__":
    main()

