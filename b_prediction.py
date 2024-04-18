from sklearn.preprocessing import StandardScaler
import numpy as np
import librosa
import keras
import pickle
from test import valeurs_plus_probables

scaler = StandardScaler()
model = keras.models.load_model("model.h5")
model.summary()

def noise(data):
    # Calcul de l'amplitude du bruit
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    
    # Ajout du bruit gaussien au signal audio
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    
    return data

# Fonction pour étirer le signal audio dans le temps
def stretch(data, rate=0.8):
    # Utilisation de la fonction time_stretch de librosa pour étirer le signal audio
    return librosa.effects.time_stretch(data, rate=rate)

# Fonction pour décaler le signal audio dans le temps
def shift(data):
    # Génération d'une plage de décalage aléatoire en millisecondes
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    
    # Décalage du signal audio
    return np.roll(data, shift_range)

# Fonction pour modifier la hauteur tonale du signal audio (modification de la hauteur)
def pitch(data, sampling_rate, pitch_factor=0.7):
    # Utilisation de la fonction pitch_shift de librosa pour modifier la hauteur tonale du signal audio
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

    # Extrayez les caractfrom main import predict_emotionéristiques audio
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
    pkl_file = open('decoder.pkl', 'rb')
    encoder = pickle.load(pkl_file) 
    pkl_file.close()

    predicted_emotion = encoder.inverse_transform(prediction)
    prob = valeurs_plus_probables(prediction)
  
    return predicted_emotion, prob[1]