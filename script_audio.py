import pyaudio
import wave
import os
def audio():
    try:

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 48000
        RECORD_SECONDS = 5
        OUTPUT_DIR = "C:/Users/naouf/Documents/1-Naoufel/1-projet/7-Hachaton/Hackaton-Gr2"
        WAVE_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "output.wav")

        p = pyaudio.PyAudio()

        for i in range(p.get_device_count()):
            print(p.get_device_info_by_index(i))

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        output=True,
                        input_device_index=3,
                        output_device_index=2,
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
        wf.setsampwidth(p.get_sample_size(FORMAT))  # Correction ici
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print("Enregistrement effectué avec succès!")
        return True
    except:
        print("Echec de l'enregistrement")
        return False

def audio_lecture():
    try:
        fichier_wav = "C:/Users/naouf/Documents/1-Naoufel/1-projet/7-Hachaton/Hackaton-Gr2/output.wav"
        print("Voici votre message")
        with wave.open(fichier_wav, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()

            # Lire les données audio
            data = wav_file.readframes(num_frames)

        # Jouer les données audio
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(sample_width),
                        channels=channels,
                        rate=frame_rate,
                        output=True)
        stream.write(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Fin du message")
        
    except:
        print("Vous devez d'abord effectuer un enregistrement avant de le l'écouter")
    return