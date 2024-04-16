import librosa
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

# Function to extract MFCCs from audio
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    return mfccs_processed

# Define a function to load data, extract features, and create labels
def load_data(data_path):
    features, labels = [], []
    for folder in os.listdir(data_path):
        for file in os.listdir(os.path.join(data_path, folder)):
            if file.lower().endswith('.wav'):
                emotion_code = file[6:8]
                emotion = emotion_map.get(emotion_code)
                file_path = os.path.join(data_path, folder, file)
                mfccs = extract_features(file_path)
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(emotion)
    return np.array(features), labels

# Emotion mapping based on RAVDESS filename conventions
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Load data
data_path = 'archive'  
features, labels = load_data(data_path)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

# Train a Support Vector Machine classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'emotion_model.pkl')
print("Model trained and saved successfully.")
