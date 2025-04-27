import streamlit as st
import objectif1
import objectif2
import objectif3
import objectif4
import objectif5
import objectif6
import objectif7
import objectif8_delay
import objectif9
import objectif10
import objectif11
import objectif12

import patientclustering
import patientassociationrules
import readmissionclassification
import burnout   
# Définition de la barre de navigation
def navigation():
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Choose Prediction or Visualization", ["Home Page","length of stay prediction", "Admission Type Prediction","predict the cluster of political subdivision","Medication-Diagnosis Associations","Forecast annual insurance costs","Predict the severity of thyroid cancer","Identify associations between medical symptoms","Predict Delay","Predict the clusters of Delay","Identify associations between Delays","BI Visualizations","predict (positive /negative)disease","Predict heart condition","Patient Clustering","Readmisssion Classification","Predict staff's burnout score","Patient Association Rules"])
    return page

# Page d'accueil
def home_page():
        st.title("Welcome to the Hospital BI and Prediction System")
        st.write("Elaborated by Cayenne")
        # Add an image to the home page
        image_url_home = "home.png"  # Replace with the URL of your image
        st.image(image_url_home, caption="Welcome to the BI and Prediction System",use_column_width=True)

def bi_visualizations_page():
    st.title("BI Visualizations")
    st.write("Click the link below to view the our report:")
    # Embed the Power BI link
    power_bi_link = "https://app.powerbi.com/reportEmbed?reportId=ea79b2ac-fe62-43f8-a831-441a2638637d&appId=61a366c5-8357-4749-9e24-56ad140ab3a4&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730"
        
        # Display the clickable link
    st.markdown(f"[Open our Report]({power_bi_link})", unsafe_allow_html=True)

    # Add an image and make it clickable to go to the Power BI report
    image_url = "powerbi.jpg"  # Replace with the URL of the image you want to use
    st.image(image_url, caption="Click to view our Report", use_column_width=True)

# Fonction principale de l'application
def main():
    page = navigation()  # Affiche la barre de navigation
    if page == "Home Page":
        home_page()  # Affiche la page d'accueil
    elif page == "length of stay prediction":
        objectif1.main()  # Appelle la fonction main de objectif1
    elif page == "Admission Type Prediction":
        objectif2.main()  # Appelle la fonction main de objectif2
    elif page == "predict the cluster of political subdivision":
        objectif3.main() 
    elif page == "Medication-Diagnosis Associations":
        objectif4.main() 
    elif page == "Forecast annual insurance costs":
        objectif5.main() 
    elif page == "Predict the severity of thyroid cancer":
        objectif6.main() 
    elif page == "Identify associations between medical symptoms":
        objectif7.main()  
    elif page == "Predict Delay":
        objectif8_delay.main()
    elif page == "Predict the clusters of Delay":
        objectif9.main()
    elif page == "Identify associations between Delays":
        objectif10.main()
    elif page == "predict (positive /negative)disease":
        objectif11.main()
    elif page == "Predict heart condition":
        objectif12.main()
    elif page == "Patient Clustering":
        patientclustering.main() 
    elif page == "Patient Association Rules":
        patientassociationrules.main() 
    elif page == "Readmisssion Classification":
        readmissionclassification.main() 
    elif page == "Predict staff's burnout score":
        burnout.main()
    elif page == "BI Visualizations":
        bi_visualizations_page() 
if __name__ == "__main__":
    main()
