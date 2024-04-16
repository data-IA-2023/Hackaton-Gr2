from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import librosa
import keras

scaler = StandardScaler()
encoder = OneHotEncoder()

model = keras.models.load_model("/Users/home/Documents/Python/Hackaton-Gr2/Emotion_Voice_Detection_Model.h5")
model.summary() 

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

def predict_emotion(audio_file):
    # Chargez le fichier audio
    data = librosa.load(audio_file, duration=2.5, offset=0.6)

    # Extrayez les caractéristiques audio
    features = extract_features(data)
    
    # Transformez les caractéristiques pour les rendre compatibles avec le modèle
    features = scaler.transform(features.reshape(1, -1))
    features = np.expand_dims(features, axis=2)
    
    # Faites une prédiction avec le modèle
    prediction = model.predict(features)
    
    # Convertissez la prédiction en émotion
    predicted_emotion = encoder.inverse_transform(prediction)
    
    return predicted_emotion[0][0]




# Utilisez cette fonction avec votre propre fichier audio
audio_file_path = "/Users/home/Documents/Python/Hackaton-Gr2/test.wav"
predicted_emotion = predict_emotion(audio_file_path)
print("Predicted emotion:", predicted_emotion)


