from b_prediction import predict_emotion, max
from d_record import audio
from e_chatbot import chat_with_bot, initial_prompt
from f_stt import speech_to_text, model_stt
import os
import streamlit as st
import hydralit_components as hc
from dotenv import load_dotenv
from hugchat import hugchat
from hugchat.login import Login
from vosk import Model

# ===== Chargement des credentials =====

load_dotenv("credentials/credentials.env")
email, password = os.environ['email'],os.environ['password']
path = "model_stt"

# ===== Présentation de la page =====

st.set_page_config(page_title="EmoBot 🤖")
st.markdown("<h1 style='text-align: center;'>Discute avec Emobot !</h1>", unsafe_allow_html=True)
st.markdown("---")

# ====== Mise en cache du chatbot =====

@st.cache_resource
def start_chatbot(email,password):
    cookie_path_dir = "./cookies/" 
    sign = Login(email, password)
    cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot

chatbot_log = start_chatbot(email, password)

@st.cache_resource
def model_stt(model_path):
    if not os.path.exists(model_path):
        print("Please download the French model.")
        return None
    else:
        model = Model(model_path)
        return model

stt_model = model_stt(path)

#===== Discussion avec le bot =====

try:
    if st.button(label="Cliquez pour discuter !"):
        with hc.HyLoader('',hc.Loaders.standard_loaders,index=5):
            audio()
            emotions = predict_emotion("output.wav")
            stt_output = speech_to_text("output.wav", stt_model)
            chatbot_discussion = chat_with_bot(chatbot_log, initial_prompt) #+ max(emotions)[0] + stt_output)
            st.sidebar.write("Emotion détéctée la plus probable")
            st.sidebar.progress(int(max(emotions)[1]), f"{max(emotions)[0].capitalize()} : {max(emotions)[1]:.3f}%")
            st.sidebar.write("Phrase entendue par Emobot !")
            st.sidebar.text_area(f"{stt_output}")
            st.write(chatbot_discussion)
except:
    st.write("ya un truc qui va pas")