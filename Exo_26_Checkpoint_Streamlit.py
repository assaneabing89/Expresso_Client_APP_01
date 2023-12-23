import streamlit as st
import numpy as np
import joblib

st.title("Application pour prédire la probabilité de désabonnement des client Expresso")
st.subheader("0 pour un client non désabonné d'expresso et 1 pour un client qui se désabonne")

# Chargement du model
model = joblib.load(filename = 'D:\Formation\Gomycode_datascience\Exo 26 Checkpoint\Exo_26_Checkpoint_V2_model.joblib')

#Définition d'une fonction d'inférence
def inférence(MONTANT, FREQUENCE_RECH, REVENUE,
       ARPU_SEGMENT, DATA_VOLUME, ON_NET, ORANGE, TIGO,
       ZONE1, ZONE2, FREQ_TOP_PACK):

    new_data = np.array( [MONTANT, FREQUENCE_RECH, REVENUE,
       ARPU_SEGMENT, DATA_VOLUME, ON_NET, ORANGE, TIGO,
       ZONE1, ZONE2, FREQ_TOP_PACK])

    pred = model.predict(new_data.reshape(1,-1))
    return pred

# l'utilisateur saisie une valeur pour chaque caractéristique de l'Analyse
MONTANT = st.number_input('MONTANT : le montant de recharge', min_value = 0.0000, value = 10.000)
FREQUENCE_RECH = st.number_input('FREQUENCE_RECH : le nombre de fois que le client a fait une recharge', min_value = 0.0000, value = 10.000)
REVENUE = st.number_input('REVENUE : revenu mensuel de chaque client', min_value = 0.0000, value = 10.000)
ARPU_SEGMENT = st.number_input('ARPU_SEGMENT : le revenu sur 90 jours/3', min_value = 0.0000, value = 10.0000)
DATA_VOLUME = st.number_input('DATA_VOLUME : le nombre de connexions', min_value = 0.0000, value = 10.0000)
ON_NET = st.number_input('ON_NET : appel inter expresso', min_value = 0.0000, value = 10.0000)
ORANGE = st.number_input('ORANGE : appel vers orange', min_value = 0.0000, value = 10.0000)
TIGO = st.number_input('TIGO : appel vers Tigo', min_value = 0.0000, value = 10.0000)
ZONE1 = st.number_input('ZONE1 : appel vers les zone1', min_value = 0.0000, value = 10.0000)
ZONE2 = st.number_input('ZONE2 : appel vers les zone2', min_value = 0.0000, value = 10.0000)
FREQ_TOP_PACK = st.number_input('FREQ_TOP_PACK : le nombre de fois que le client a activé les packages top pack', min_value = 0.0000, value = 10.0000)

#creation d'un bouton prédict
if st.button('Predict'):
     prediction = inférence(MONTANT, FREQUENCE_RECH, REVENUE,
       ARPU_SEGMENT, DATA_VOLUME, ON_NET, ORANGE, TIGO,
       ZONE1, ZONE2, FREQ_TOP_PACK)
     résultat = 'la probabilité de désabonnement des clients est de '+ str (prediction[0])
     st.success(résultat)




