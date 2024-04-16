from hugchat import hugchat
from hugchat.login import Login
from dotenv import load_dotenv
import os
load_dotenv()

email, password = os.environ['email'],os.environ['password']

def start_chatbot(email,password ):
    cookie_path_dir = "./cookies/" 
    sign = Login(email, password)
    cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)
    
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    print(' Welcome to ChatPAL. Let\'s talk my friend ! ')
    print('\'q\' or \'quit\' to exit')
    print('\'c\' or \'change\' to change conversation')
    print('\'n\' or \'new\' to start a new conversation')
    return chatbot

def chat_with_bot(chatbot, msg, emotion):
    msg="{prend en compte l'emotion} "+emotion+' : '+msg
    print(msg)
    if msg.lower() == '':
        pass
    elif msg.lower() in ['q', 'quit']:
        return None
    elif msg.lower() in ['c', 'change']:
        print('Choose a conversation to switch to:')
        print(chatbot.get_conversation_list())
    elif msg.lower() in ['n', 'new']:
        print('Clean slate!')
        id = chatbot.new_conversation()
        chatbot.change_conversation(id)
    else:
        return chatbot.chat(msg)
    
chatbot = start_chatbot(email,password)

print(chat_with_bot(chatbot, "donne moi vite le meilleur restau de Tours",'furieux'))