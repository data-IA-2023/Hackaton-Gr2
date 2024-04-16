import librosa
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import os
from sklearn.preprocessing import LabelEncoder
import resampy

os.environ['TF_ENABLE_ONEDNN_OPTS'] = 0

# Charger les variables d'environnement à partir du fichier .env


# Function to extract features from audio
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Function to load audio data and labels
def load_data(data_folder="C:/Users/naouf/Documents/1-Naoufel/1-projet/7-Hachaton/Hackaton-Gr2/statics/archive"):
    audio_data = []
    labels = []

    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                # Extract emotion label from the filename
                emotion = file.split("-")[2]
                labels.append(emotion)
                audio_data.append(file_path)

    return audio_data, labels

# Load audio data and labels
audio_data, labels = load_data(data_folder="C:/Users/naouf/Documents/1-Naoufel/1-projet/7-Hachaton/Hackaton-Gr2/statics/archive")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(audio_data, labels, test_size=0.2, random_state=42)



# Feature extraction
X_train = np.array([extract_features(x) for x in X_train])
X_test = np.array([extract_features(x) for x in X_test])




# Function to encode emotion labels
def encode_labels(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return to_categorical(encoded_labels)
# Encoding labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Define and compile the LSTM model
model = Sequential()
model.add(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(units=y_train_encoded.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
model.fit(X_train, y_train_encoded, epochs=50, batch_size=32, validation_split=0.2)


# Model evaluation
loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')