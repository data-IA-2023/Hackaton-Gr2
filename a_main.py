import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from b_functions import extract_features
import numpy as np
import librosa


model = keras.models.load_model("model_save/Emotion_Voice_Detection_Model.h5")
scaler = StandardScaler()
encoder = OneHotEncoder()

def predict_emotion(audio_file):
    # Chargez le fichier audio
    data = librosa.load(audio_file, duration=2.5, offset=0.6)[0]
    
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
audio_file_path = "ressource_audio/custom/test_voice.mp3"
predicted_emotion = predict_emotion(audio_file_path)
print("Predicted emotion:", predicted_emotion)

