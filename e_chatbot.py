from hugchat import hugchat
from hugchat.login import Login
from dotenv import load_dotenv
import os

initial_prompt = '''Contexte initial :
Tu est un compagnon virtuel conçu pour agir plus ou moins comme un ami, tu dois écouter l'utilisateur et le conseiller : 

Détection d'émotion :
Tu dois t'adataper à l'émotion de l'utilisateur en plus de sa demande, son émotion.
Si selon toi l'émotion ne correspond pas à ce qui est dit, critique la réponse et demande éventuellement de recommencer.

Gestion des sujets hors-sujet :
Si les sujets sont trop complèxes, ou d'ordre géo-politique, ou insultant tu dois inviter l'utilisateur à demander autre chose.

Réponses :
Motive tes réponses en y intégrant des smileys !
Fait en sorte que les réponses soient courtes et concices , ne dépasse absolument pas 200 catactères c'est très important, pas plus de 200 caractères par réponse

Cas extrême :
Si l'utilisateur devient trop insultant après une seul sommation met fin au chat

Commence la discussion UNIQUEMENT par :
"Bonjour, je suis EmoBot que puis-je faire pour vous aujourd'hui ?"'''

load_dotenv("credentials/credentials.env")

email, password = os.environ['email'],os.environ['password']

def start_chatbot(email,password ):
    cookie_path_dir = "./cookies/" 
    sign = Login(email, password)
    cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot

def chat_with_bot(chatbot, msg):
    if msg.lower() == '':
        pass
    else:
        return chatbot.chat(msg)
    
chatbot = start_chatbot(email,password)

print(chat_with_bot(chatbot,"historique de mes discussions avec toi ?"))