
from b_prediction import predict_emotion
from d_record import audio
from e_chatbot import start_chatbot
import os
import keras
import streamlit as st
import hydralit_components as hc

st.set_page_config(page_title="EmoBot 🤖")

st.markdown("<h1 style='text-align: center;'>Discute avec Emobot !</h1>", unsafe_allow_html=True)
st.markdown("---")



#===== Enregistrement de l'audio =====

try:
    if st.button(label="Cliquez pour discuter !"):
        with hc.HyLoader('',hc.Loaders.standard_loaders,index=5):
            audio()


except:
    print("sah c'est domaj")