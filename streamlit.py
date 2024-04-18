import streamlit as st
from script_audio import audio
from script_audio import audio_lecture
from modele import all
import os
import keras



st.markdown("<h1 style='text-align: center;'><u>Reconnaissance vocale d'émotions</u></h1>", unsafe_allow_html=True)


try:
    
    if st.button(label="Appuyez-ici pour débuter l'enregistrement"):
        audio()
except:
    st.write("Erreur de l'enregistrement")


try:
    path = "C:/Users/naouf/Documents/1-Naoufel/1-projet/7-Hachaton/Hackaton-Gr2/output.wav"
    if os.path.exists(path):
        try:
            if st.button("Appuyer ici pour lancer l'analyse de votre voix"):

                predicted_emotion = all(path)

                print(predicted_emotion)

                st.write(predicted_emotion)
        except:
            st.write("Erreur du modele")

        if st.button(label="Appuyez-ici pour entendre votre enregistrement"):
            st.audio(path, format='audio/wav')
        try:
            if st.button(label="Appuyez-ici pour supprimer l'enregistrement"):
                os.remove(path)
        except:
            st.write("Erreur de la supression")
except:
    st.write("Erreur lors du chargement de l'audio")




















