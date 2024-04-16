from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import librosa
import keras
import pyaudio
import wave
import os
import pickle
import pandas as pd

scaler = StandardScaler()


model = keras.models.load_model("/Users/home/Documents/Python/Hackaton-Gr2/Emotion_Voice_Detection_Model.h5")
model.summary() 

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features(datatuple):
    # ZCR
    data = datatuple[0]
    sample_rate = datatuple[1]
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result



def get_features(datatuple):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data = datatuple[0]
    sample_rate = datatuple[1]
    print("without augmentation")
    res1 = extract_features(datatuple)
    result = np.array(res1)
    print("data with noise")
    noise_data = noise(data)
    res2 = extract_features((noise_data, sample_rate))
    result = np.vstack((result, res2)) # stacking vertically
    
    print("data with stretching and pitching")
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features((data_stretch_pitch, sample_rate))
    result = np.vstack((result, res3)) # stacking vertically
    
    return result

def predict_emotion(audio_file):
    # Chargez le fichier audio
    data = librosa.load(audio_file, duration=2.5, offset=0.6)

    # Extrayez les caractéristiques audio
    X = []
    feature = get_features(data)
    for ele in feature:
        X.append(ele)
    features = X
    
    # Transformez les caractéristiques pour les rendre compatibles avec le modèle
    features = scaler.fit_transform(features)
    features = np.expand_dims(features, axis=2)
    
    # Faites une prédiction avec le modèle
    prediction = model.predict(features)
    
    # Convertissez la prédiction en émotion
    pkl_file = open('Departure_encoder.pkl', 'rb')
    encoder = pickle.load(pkl_file) 
    pkl_file.close()

    predicted_emotion = encoder.inverse_transform(prediction)

    
    return predicted_emotion[0][0]

def audio():

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 48000
    RECORD_SECONDS = 5
    OUTPUT_DIR = "/static"
    WAVE_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "output.wav")

    p = pyaudio.PyAudio()

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


# Utilisez cette fonction avec votre propre fichier audio
audio_file_path = "/Users/home/Documents/Python/Hackaton-Gr2/static/test.wav"
predicted_emotion = predict_emotion(audio_file_path)
print("Predicted emotion:", predicted_emotion)


