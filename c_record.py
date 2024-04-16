import pyaudio
import wave

def record_audio(filename, duration=5, chunk=1024, channels=1, sample_rate=44100):
    audio = pyaudio.PyAudio()

    # Open a new stream for recording
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)

    print("Enregistre...")

    frames = []

    # Record audio in chunks
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Enregistrement terminé.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

if __name__ == "__main__":
    filename = "/ressource_audio/custom/recorded_audio.wav"  # Change this to your desired filename
    duration = 5  # Change this to adjust the duration of the recording
    record_audio(filename, duration)