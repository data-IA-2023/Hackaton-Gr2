import streamlit as st
from script_audio import audio
from script_audio import audio_lecture
import os
import keras

try:
    load = keras.models.load_model("C:/Users/naouf/Documents/1-Naoufel/1-projet/7-Hachaton/Hackaton-Gr2/modele/model_final_1.h5")
    load.summary()
except:
    print("Le modele n'a pas pu etre chargé")

st.markdown("<h1 style='text-align: center;'><u>Reconnaissance vocale d'émotions</u></h1>", unsafe_allow_html=True)


try:
    
    if st.button(label="Appuyez-ici pour débuter l'enregistrement"):
        audio()
except:
    st.write("Erreur de l'enregistrement")


try:
    path = "C:/Users/naouf/Documents/1-Naoufel/1-projet/7-Hachaton/Hackaton-Gr2/output.wav"
    if os.path.exists(path):
        if st.button(label="Appuyez-ici pour entendre votre enregistrement"):
            st.audio(path, format='audio/wav')
        try:
            if st.button(label="Appuyez-ici pour supprimer l'enregistrement"):
                os.remove(path)
        except:
            st.write("Erreur de la supression")
except:
    st.write("Erreur lors du chargement de l'audio")






audio_file_path = "/Users/home/Documents/Python/Hackaton-Gr2/ressource_audio/custom/output.wav"
predicted_emotion = predict_emotion(audio_file_path)
print("Predicted emotion:", predicted_emotion)

















