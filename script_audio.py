import pyaudio
import wave
import os
def audio():

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
    return 