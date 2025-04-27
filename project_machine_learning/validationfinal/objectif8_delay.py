import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
# import test
# from mlxtend.frequent_patterns import apriori, association_rules
# Charger le modèle
rfmodel = joblib.load('delay/random_forest_model.pkl')
rgscaler = joblib.load('delay/scaler_model.pkl')
label_data = pd.read_csv('delay/unique_labels.txt', delimiter=',')  # Change delimiter as needed (e.g., '\t' for tab, ',' for comma)
wardid_data = pd.read_csv('delay/unique_wardid.txt', delimiter=',')  # Change delimiter as needed (e.g., '\t' for tab, ',' for comma)

# apscaler = StandardScaler()



# kmean_model = joblib.load('delay/kmean_model.pkl')
# print(data)
# Initialisation des encodeurs
label_encoder= LabelEncoder()
# label_encoder_description = LabelEncoder()
# label_encoder_discharge_location = LabelEncoder()
event_encoder = OrdinalEncoder(categories=[['chartevent','datetimeevent','inputevent_cv1','inputevent_mv1','microbiologyevent','procedureevent','outputevent']])
# linksto_encoder = OrdinalEncoder(categories=[['chartevents','datetimeevents','inputevents_cv','inputevents_mv','microbiologyevents','procedureevents_mv','outputevents']])
#['chartevents','datetimeevents','inputevents_cv','inputevents_mv','microbiologyevents','procedureevents_mv','outputevents']

label_encoder.fit(label_data)



# Fonction pour encoder les colonnes
def encode_label(label):
    return label_encoder.transform([label])[0]


# Fonction pour encoder les colonnes
def encode_event(event):
    return event_encoder.fit_transform([[event]]) #.transform([event])[0]


# # Fonction pour encoder les colonnes
# def encode_linksto(event):
#     return event_encoder.fit_transform([[event]])


# Fonction pour la prédiction
def predict_delay(features):
    print("testing")
    print(features)
    scaled_features = rgscale(features)
    print("testing")
    print(scaled_features)
    prediction = rfmodel.predict(scaled_features[:,[0,2,6,7,8]])
    return prediction[0]


def rgscale(features):
    print("testing scaling B")
    print(features)
    feat = pd.DataFrame([features],columns=['label','linksto','first_wardid','last_wardid','eventtype','year','month','day','timestamp'])
    print("testing  scaling A")
    print(feat)
    return rgscaler.transform(feat)



# Page de prédiction
def prediction_page():
    st.title("Prediction of the Event's Delay.")

    item_label = st.selectbox("Item Label", options=label_data.sort_values(by='labels',ascending=True))
    first_wardid = st.selectbox('First Ward Id Number',options=wardid_data.sort_values(by='wardid',ascending=True))
    last_wardid = st.selectbox('Last Ward Id Number',options=wardid_data.sort_values(by='wardid',ascending=True))

    eventtype= st.selectbox('Event Type',options=['chartevent','datetimeevent','inputevent_cv1','inputevent_mv1','microbiologyevent','procedureevent','outputevent'])
    linksto= st.selectbox('Linksto',options=['chartevent','datetimeevent','inputevent_cv1','inputevent_mv1','microbiologyevent','procedureevent','outputevent'])
    
    date = st.date_input("Date of the Event")
    hours=st.number_input("Hour", min_value=0, step=1,max_value=24)
    minutes =st.number_input("Minutes", min_value=0, step=1,max_value=59)
    encoded_label = encode_label(item_label)
    encoded_event = encode_event(eventtype)
    encoded_linksto = encode_event(linksto)

    datetime = pd.to_datetime(f"{date} {hours}:{minutes}")
    

    features = [
        encoded_label,
        encoded_linksto,
        first_wardid,
        last_wardid,
        encoded_event,
        datetime.year,
        datetime.month,
        datetime.day,
        datetime.timestamp(),
    ]
    
    
    if st.button("Predict"):
        result = predict_delay(features)
        st.subheader("Result of the Prediction :")
        st.success(result)
        # if st.button('pass values to clustering'):
        #     clustering_page(result,item_label,first_wardid,last_wardid,eventtype,linksto,date,hours,minutes)



# Fonction principale
def main():
    prediction_page()
    

# Lancer l'application
if __name__ == "__main__":
    # Style personnalisé
    main()