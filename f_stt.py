from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import wave
import json
import io
import os

def model_stt(model_path):
    if not os.path.exists(model_path):
        print("Please download the French model.")
        return None
    else:
        model = Model(model_path)
        return model

def speech_to_text(output,model):

    # Load and convert the MP3 file
    audio = AudioSegment.from_file(output, format="mp3")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2) 
    audio_bytes = io.BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)

    # Use wave to read the converted audio
    wf = wave.open(audio_bytes, "rb")
    rec = KaldiRecognizer(model, wf.getframerate(), '{"lang": "fr"}')  # Fix JSON format
    
    # Recognize speech
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            data = json.loads(rec.Result())
            return data["text"]