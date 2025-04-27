import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('class_nour.pkl')

def predict_disease(features):
    """
    Predict disease positivity or negativity based on input features.
    Args:
        features (numpy array): The input features for the model.
    Returns:
        str: "Positive" if prediction is -1, otherwise "Negative".
    """
    try:
        prediction = model.predict(features)
        return "Positive" if prediction[0] == -1 else "Negative"
    except Exception as e:
        return f"Error: {e}"

def main():
    """
    Main function to render the Streamlit application.
    """
    # App title
    st.title("Disease (Positive/Negative) Prediction")

    # Input fields for user data
    st.header("Enter the measures")

    # Input for Disease
    disease = st.selectbox(
        "Select the Disease",
        ["Asthma", "Stroke", "Osteoporosis", "Hypertension", "Diabetes", "Migraine"],
    )
    # Map disease to a numerical value
    disease_mapping = {
        "Asthma": 6,
        "Stroke": 101,
        "Osteoporosis": 77,
        "Hypertension": 51,
        "Diabetes": 32,
        "Migraine": 69,
    }
    disease_numeric = disease_mapping[disease]

    # Binary features with Yes/No options
    fever = st.selectbox("Fever", ["No", "Yes"], help="Select 'Yes' if the patient has fever, otherwise 'No'")
    cough = st.selectbox("Cough", ["No", "Yes"], help="Select 'Yes' if the patient has cough, otherwise 'No'")
    fatigue = st.selectbox("Fatigue", ["No", "Yes"], help="Select 'Yes' if the patient experiences fatigue, otherwise 'No'")
    difficulty_breathing = st.selectbox("Difficulty Breathing", ["No", "Yes"], help="Select 'Yes' if the patient has difficulty breathing, otherwise 'No'")
    
    # Convert Yes/No to numeric (No = 0, Yes = 1)
    fever_numeric = 1 if fever == "Yes" else 0
    cough_numeric = 1 if cough == "Yes" else 0
    fatigue_numeric = 1 if fatigue == "Yes" else 0
    difficulty_breathing_numeric = 1 if difficulty_breathing == "Yes" else 0

    # Numeric input for age
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

    # Gender input with Male/Female options
    sex = st.selectbox("Gender", ["Female", "Male"], help="Select 'Female' or 'Male'")
    gender_numeric = 0 if sex == "Female" else 1

    # Blood Pressure with categories
    blood_pressure = st.selectbox("Blood Pressure", ["Low", "Normal", "High"], help="Select the patient's blood pressure level")

    # Cholesterol Level with categories
    cholesterol_level = st.selectbox("Cholesterol Level", ["Low", "Normal", "High"], help="Select the patient's cholesterol level")

    # Convert categorical inputs to numeric values
    blood_pressure_mapping = {"Low": 1, "Normal": 2, "High": 0}
    cholesterol_level_mapping = {"Low": 1, "Normal": 2, "High": 0}
    blood_pressure_numeric = blood_pressure_mapping[blood_pressure]
    cholesterol_level_numeric = cholesterol_level_mapping[cholesterol_level]

    # Create feature array (ensure it's 2D)
    features = np.array([[disease_numeric, fever_numeric, cough_numeric, fatigue_numeric, 
                          difficulty_breathing_numeric, age, gender_numeric, 
                          blood_pressure_numeric, cholesterol_level_numeric]])

    # Predict button
    if st.button("Predict"):
        result = predict_disease(features)
        if "Error" in result:
            st.error(result)
        else:
            st.success(f"The result is: {result}")

# Run the app
if __name__ == "__main__":
    main()
