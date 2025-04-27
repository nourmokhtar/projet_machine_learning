import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder 






label_data = pd.read_csv('delay/unique_labels.txt', delimiter=',')  # Change delimiter as needed (e.g., '\t' for tab, ',' for comma)
wardid_data = pd.read_csv('delay/unique_wardid.txt', delimiter=',')  # Change delimiter as needed (e.g., '\t' for tab, ',' for comma)





kmean_model = joblib.load('delay/kmean_model.pkl')
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




# Fonction pour la prédiction
def cluster_predict(features):
    feat = pd.DataFrame([features],columns=['value','label','linksto','first_wardid','last_wardid','eventtype','year','month','day','timestamp'])
    prediction = kmean_model.predict(feat)
    return prediction[0]



def clustering_page():
    st.title("The Prediction of Clusters for Event's Delay.")

    # Sorting label data by 'labels'
    sorted_label_data = label_data.sort_values(by='labels', ascending=True)
    
    # Pre-fill the selectbox for 'item_label' if features is not empty
    clitem_label = st.selectbox(
        "Item label",
        options=sorted_label_data['labels'],
        # index=sorted_label_data[sorted_label_data['labels'] == label].index[0] if label!=''  else 0
    )

    # Sorting ward id data by 'wardid'
    sorted_wardid_data = wardid_data.sort_values(by='wardid', ascending=True)
    
    # Pre-fill the selectboxes for 'first_wardid' and 'last_wardid' if features exist
    clfirst_wardid = st.selectbox(
        'First Ward ID Number',
        options=sorted_wardid_data['wardid'],
        # index=sorted_wardid_data[sorted_wardid_data['wardid'] == first_wardid].index[0] if first_wardid!=0 else 0
    )
    
    cllast_wardid = st.selectbox(
        'Last Ward ID Number',
        options=sorted_wardid_data['wardid'],
        # index=sorted_wardid_data[sorted_wardid_data['wardid'] == last_wardid].index[0] if last_wardid!=0 else 0
    )

    # Event type and 'linksto' dropdowns
    cleventtype = st.selectbox(
        'Event Type',
        options=['chartevent', 'datetimeevent', 'inputevent_cv1', 'inputevent_mv1', 'microbiologyevent', 'procedureevent', 'outputevent'],
        # index=['chartevent', 'datetimeevent', 'inputevent_cv1', 'inputevent_mv1', 'microbiologyevent', 'procedureevent', 'outputevent'].index(eventtype) if eventtype!='' else 0
    )
    
    cllinksto = st.selectbox(
        'Linksto',
        options=['chartevent', 'datetimeevent', 'inputevent_cv1', 'inputevent_mv1', 'microbiologyevent', 'procedureevent', 'outputevent'],
        # index=['chartevent', 'datetimeevent', 'inputevent_cv1', 'inputevent_mv1', 'microbiologyevent', 'procedureevent', 'outputevent'].index(linksto) if linksto!='' else 0
    )
    
    # Date and time selection (with pre-filled values if available in features)
    cldate = st.date_input("Date of the Event",
        # value=date if date==0 else pd.to_datetime("2024-01-01")
    )
    clhours = st.number_input("Hour", min_value=0, step=1, max_value=24, 
        # value=hours
    )
    clminutes = st.number_input("Minutes", min_value=0, step=1, max_value=59, 
        # value=minutes
    )
    
    delay = st.number_input("Delay", min_value=0, 
        # value=value
    )
    
    # Encoding the selected values
    encoded_label = encode_label(clitem_label)
    encoded_event = encode_event(cleventtype)
    encoded_linksto = encode_event(cllinksto)

    # Combining the date and time into a datetime object
    cldatetime = pd.to_datetime(f"{cldate} {clhours}:{clminutes}")
    
    # # Printing for debugging
    # print('Test')
    # st.write(f"Selected Date and Time: {cldatetime}")

    # You can add further code for predictions or other functionalities here.
    features = [
        delay,
        encoded_label,
        encoded_linksto,
        clfirst_wardid,
        cllast_wardid,
        encoded_event,
        cldatetime.year,
        cldatetime.month,
        cldatetime.day,
        cldatetime.timestamp(),
    ]
    
    if st.button("Predict"):
        result = cluster_predict(features)
        st.subheader("Résult of the Prediction :")
        if ( result ==0):
            result_text =' most likely middle day of summer'
        if ( result ==1):
            result_text =' wardid around 50 , early hours '
        if ( result ==2):
            result_text ='early hours , wardid around 50  , winter time '
        if ( result ==3):
            result_text ='summer time , end of day , very close by wards ? '
        if ( result ==4):
            result_text ='end of year , far wards '
        if ( result ==5):
            result_text ='close wards , late day hours  '
        st.success(result_text)


# Fonction principale
def main():
    clustering_page()
    

# Lancer l'application
if __name__ == "__main__":
    # Style personnalisé
    main()