
from b_prediction import predict_emotion
from d_record import audio
import os
import keras
import streamlit as st

st.set_page_config(page_title="EmoBot 🤖")

st.markdown("<h1 style='text-align: center;'><u>Reconnaissance vocale d'émotions</u></h1>", unsafe_allow_html=True)

#===== Enregistrement de l'audio =====

try:
    if st.button(label="Appuyez-ici pour débuter l'enregistrement"):
        audio()
        
except:
    st.write("Erreur de l'enregistrement")


try:
    path = "output.wav"
    if os.path.exists(path):
        try:
            if st.button("Discuter avec EmoBot"):
                predicted_emotion = predict_emotion(path)
                print(predicted_emotion)
                st.write(predicted_emotion)

        except Exception as e :
            st.write(f"Erreur du modele,{e}")

except:
    st.write("Erreur lors du chargement de l'audio")