import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

clustering_model_path = r"clustering_KMEANS.pkl"  # Update with your KMeans clustering model
clustering_model = joblib.load(clustering_model_path)
# Initialize encoders for clustering model
label_encoders = {
    "Gender": LabelEncoder(),
    "Smoking": LabelEncoder(),
    "Hx Smoking": LabelEncoder(),
    "Hx Radiotherapy": LabelEncoder(),
    "Thyroid Function": LabelEncoder(),
    "Physical Examination": LabelEncoder(),
    "Adenopathy": LabelEncoder(),
    "Pathology": LabelEncoder(),
    "Focality": LabelEncoder(),
    "Risk": LabelEncoder(),
    "T": LabelEncoder(),
    "N": LabelEncoder(),
    "M": LabelEncoder(),
    "Recurred": LabelEncoder(),
}
ordinal_encoder = OrdinalEncoder()
# Fit encoders with predefined categories
label_encoders["Gender"].fit(["M", "F"])
label_encoders["Smoking"].fit(["Yes", "No"])
label_encoders["Hx Smoking"].fit(["Yes", "No"])
label_encoders["Hx Radiotherapy"].fit(["Yes", "No"])
label_encoders["Thyroid Function"].fit(['Euthyroid', 'Clinical Hyperthyroidism', 'Clinical Hypothyroidism', 'Subclinical Hyperthyroidism', 'Subclinical Hypothyroidism'])
label_encoders["Physical Examination"].fit(["Single nodular goiter-left", "Multinodular goiter", "Single nodular goiter-right", "Normal", "Diffuse goiter"])
label_encoders["Adenopathy"].fit(["No", "Right", "Extensive", "Left", "Bilateral", "Posterior"])
label_encoders["Pathology"].fit(["Micropapillary", "Papillary", "Follicular", "Hurthel cell"])
label_encoders["Focality"].fit(["Uni-Focal", "Multi-Focal"])
label_encoders["Risk"].fit(["Low", "Intermediate", "High"])
label_encoders["T"].fit(["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"])
label_encoders["N"].fit(["N0", "N1b", "N1a"])
label_encoders["M"].fit(["M0", "M1"])
label_encoders["Recurred"].fit(["Yes", "No"])
ordinal_encoder.fit([["Indeterminate"], ["Excellent"], ["Structural Incomplete"], ["Biochemical Incomplete"]])

# Function to encode user inputs for clustering model
def encode_inputs(data):
    for col, encoder in label_encoders.items():
        if col in data:
            data[col] = encoder.transform([data[col]])[0]
    if "Response" in data:
        data["Response"] = ordinal_encoder.transform([[data["Response"]]])[0][0]
    return data

# Prediction function for clustering model
def predict_cluster(features, model):
    try:
        return model.predict([features])[0]  # Use predict instead of fit_predict
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None
    
    
def main():
    st.title("Cluster Prediction for Thyroid Cancer")
    st.write("Provide the following details to predict the patient's cluster.")

    # Collect user inputs (same as before)
    gender = st.selectbox("Gender", options=["M", "F"])
    smoking = st.selectbox("Smoking Status", options=["Yes", "No"])
    hx_smoking = st.selectbox("History of Smoking", options=["Yes", "No"])
    hx_radiotherapy = st.selectbox("History of Radiotherapy", options=["Yes", "No"])
    thyroid = st.selectbox("Thyroid Function", options=['Euthyroid','Clinical Hyperthyroidism','Clinical Hypothyroidism','Subclinical Hyperthyroidism','Subclinical Hypothyroidism'])
    physical_exam = st.selectbox(
        "Physical Examination",
        options=["Single nodular goiter-left", "Multinodular goiter", "Single nodular goiter-right", "Normal", "Diffuse goiter"],
    )
    adenopathy = st.selectbox(
        "Adenopathy",
        options=["No", "Right", "Extensive", "Left", "Bilateral", "Posterior"],
    )
    pathology = st.selectbox("Pathology", options=["Micropapillary", "Papillary", "Follicular", "Hurthel cell"])
    focality = st.selectbox("Focality", options=["Uni-Focal", "Multi-Focal"])
    risk = st.selectbox("Risk", options=["Low", "Intermediate", "High"])
    t = st.selectbox("T", options=["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"])
    n = st.selectbox("N", options=["N0", "N1b", "N1a"])
    m = st.selectbox("M", options=["M0", "M1"])
    recurred = st.selectbox("Recurred", options=["Yes", "No"])
    response = st.selectbox(
        "Response Stage",
        options=["Indeterminate", "Excellent", "Structural Incomplete", "Biochemical Incomplete"],
    )

    # Add age input
    age = st.number_input("Age", min_value=0, max_value=120, step=1, value=30)  # default age is 30

    # Compile user input into a dictionary
    user_input = {
        "Gender": gender,
        "Smoking": smoking,
        "Hx Smoking": hx_smoking,
        "Hx Radiotherapy": hx_radiotherapy,
        "Thyroid Function": thyroid,
        "Physical Examination": physical_exam,
        "Adenopathy": adenopathy,
        "Pathology": pathology,
        "Focality": focality,
        "Risk": risk,
        "T": t,
        "N": n,
        "M": m,
        "Recurred": recurred,
        "Response": response,
        "Age": age,  # Added Age
    }

    # Encode user inputs
    encoded_data = encode_inputs(user_input)

    if encoded_data:
        # Convert encoded data into a feature list
        features = list(encoded_data.values())

        # Predict cluster when button is clicked
        if st.button("Predict Cluster"):
            cluster = predict_cluster(features, clustering_model )
            if cluster is not None:
                if cluster == 1:
                    st.success("Patient with severe Thyroid cancer Stage.")
                elif cluster == 0:
                    st.success("Patient with moderate Thyroid cancer Stage.")
                else:
                    st.warning(f"Predicted Cluster: {cluster} — This cluster indicates an outlier according to the KMeans model.")
                    
if __name__ == "__main__":
    main()
