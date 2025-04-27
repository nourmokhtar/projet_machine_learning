import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Function to handle data loading, preprocessing, and clustering
def load_and_cluster_data(file_name):
    """
    Load and preprocess the data, then perform clustering.
    """
    # Load the data
    df = pd.read_csv(file_name)
    df.drop(columns=['target'], errors='ignore', inplace=True)  # Drop target column if it exists
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    # Perform clustering
    cah = AgglomerativeClustering(n_clusters=2, linkage='ward')  # Using 2 clusters
    labels = cah.fit_predict(data_scaled)
    return pd.DataFrame(data_scaled, columns=df.columns), df.columns, scaler, cah, labels

# Function to map feature names to user-friendly labels
def map_feature_name(feature_name):
    """
    Map the feature name to a user-friendly label.
    """
    feature_mapping = {
        "age": "Age",
        "sex": "Gender",
        "cp": "Chest Pain",
        "trestbps": "Resting Blood Pressure",
        "chol": "Serum Cholesterol",
        "fbs": "Fasting Blood Sugar in mg/dl",
        "restecg": "Resting Electrocardiographic Results",
        "thalach": "Maximum Heart Rate Achieved",
        "exang": "Exercise Induced Angina",
        "oldpeak": "Depression Induced by Exercise Relative to Rest",
        "slope": "Slope of the Peak Exercise ST Segment",
        "ca": "Number of Major Vessels Colored by Fluoroscopy",
        "thal": "Thalassemia"
    }
    return feature_mapping.get(feature_name.lower(), feature_name)

# Main function to run the Streamlit app
def main():
    st.title("Heart Conditions Clustering")

    # Load and preprocess data from the local file
    file_name = "heart.csv"  # Ensure the file is in the same directory
    try:
        data_scaled, feature_names, scaler, model, labels = load_and_cluster_data(file_name)
        
    except FileNotFoundError:
        st.sidebar.error(f"Dataset file '{file_name}' not found. Please ensure it's in the same directory.")
        st.stop()

    # Interactive form for individual prediction
    st.subheader("Predict Heart Condition for Patient")
    with st.form("prediction_form"):
        input_data = []
        for feature in feature_names:
            # Map feature names to user-friendly labels
            feature_label = map_feature_name(feature)
            
            # Handle gender input separately (for "sex" feature)
            if feature.lower() == "sex":
                sex = st.selectbox(f"Select {feature_label}:", ["Female", "Male"])
                input_data.append(0 if sex == "Female" else 1)
            
            # Handle chest pain input separately (for "cp" feature)
            elif feature.lower() == "cp":
                chest_pain = st.selectbox(
                    f"Select {feature_label}:", 
                    ["No Chest Pain", "Medium Pain", "High Pain","severe pain"]
                )
                chest_pain_mapping = {"No Chest Pain": 0, "Medium Pain": 1, "High Pain": 2,"severe pain": 3}
                input_data.append(chest_pain_mapping[chest_pain])
            
            else:
                value = st.number_input(f"Enter value for {feature_label}:", value=0)
                input_data.append(value)
        
        submitted = st.form_submit_button("Predict Patient Heart Condition")
        if submitted:
            # Scale the user input
            scaled_input = scaler.transform([input_data])
            # Append the scaled input to the dataset
            augmented_data = np.vstack([data_scaled, scaled_input])
            # Fit the model to the augmented dataset
            labels = model.fit_predict(augmented_data)
            # Return the cluster of the last data point (user input)
            cluster = labels[-1]

            # Display results based on the cluster
            if cluster == 1:
                st.success(f"The individual belongs to Cluster: {cluster}")
                st.info("This cluster represents individuals with more severe or potentially reversible cardiovascular conditions.")
            elif cluster == 0:
                st.success(f"The individual belongs to Cluster: {cluster}")
                st.info("This cluster represents individuals with higher chronic or longstanding heart conditions.")
            else:
                st.success(f"The individual belongs to Cluster: {cluster}")
                st.info("This cluster represents individuals with less severe or well-managed conditions.")

# Run the main function
if __name__ == "__main__":
    main()
