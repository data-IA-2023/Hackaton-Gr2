import pyaudio
import wave
import os

def audio():
    try:
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1  
        RATE = 48000
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = os.path.join("output.wav")

        p = pyaudio.PyAudio()

        # Utiliser les index des périphériques par défaut
        input_device_index = p.get_default_input_device_info()['index']
        output_device_index = p.get_default_output_device_info()['index']

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        output=True,
                        input_device_index=input_device_index,
                        output_device_index=output_device_index,
                        frames_per_buffer=CHUNK)

        print("Enregistrement en cours...")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("Enregistrement terminé.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print("Enregistrement effectué avec succès!")
        return True
    except Exception as e:
        print("Échec de l'enregistrement:", e)
        return False
    
audio()