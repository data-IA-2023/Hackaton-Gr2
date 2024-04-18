import requests
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
import re

def text_to_speech(text, lang="fr", speed="normal" ):
    url = "https://text-to-speech-api.p.rapidapi.com/text-to-speech"
    payload = {
        "text": text,
        "lang": lang,
        "speed": speed

    }
    headers = {
    "content-type": "application/json",
    "X-RapidAPI-Key": "341580146bmsh6d7bf77c4ad811ep1d8340jsn825f96cd7f0a",
    "X-RapidAPI-Host": "text-to-speach-api.p.rapidapi.com"
}

    # Sending a POST request to the API
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        # Read the binary response content
        audio_content = BytesIO(response.content)
        # Use pydub to play the audio
        song = AudioSegment.from_file(audio_content, format="mp3")
        play(song)
    else:
        print("Failed to get audio:", response.text)

def clean_emoji(text):
    text_clean = re.sub(r'[^\w\s,]','', text)
    return text_clean

def split_text(text, max_length=200):
    text_chunks = []
    while text :
        chunk = text[:max_length]
        text = text[max_length:]
        text_chunks.append(chunk)

    return text_chunks