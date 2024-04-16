import requests
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

def text_to_speech(text, lang="fr", speed="slow" ):
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

# Example usage
text = "Paolo mange tes morts"
text_to_speech(text)



#install "sudo apt-get install ffmpeg " for linux 
# install "brew install ffmpeg" for mac 
# windows donload it from https://ffmpeg.org/download.html 