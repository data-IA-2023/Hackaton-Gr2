# import librosa
# import numpy as np
# from sklearn.svm import SVC
# import joblib  # To load a pre-trained model

# # Function to extract MFCCs from an audio file
# def extract_features(file_path):
#     try:
#         audio, sample_rate = librosa.load(file_path, sr=None)
#         mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
#         mfccs_processed = np.mean(mfccs.T, axis=0)
#     except Exception as e:
#         print("Error encountered while parsing file: ", file_path)
#         return None, None
#     return mfccs_processed

# # Load pre-trained model (Assuming the model is saved as 'emotion_model.pkl')
# model = joblib.load('emotion_model.pkl')

# # Input from user
# file_path = input("Enter the path to the voice file: ")

# # Extract features and predict emotion
# features = extract_features(file_path)
# if features is not None:
#     features = np.array([features])  # Make it compatible for prediction
#     predicted_emotion = model.predict(features)
#     print("Predicted Emotion:", predicted_emotion[0])
# else:
#     print("Feature extraction failed.")


import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.svm import SVC
import joblib  # To load a pre-trained model

# Function to extract MFCCs from audio data
def extract_features(audio, sample_rate):
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while processing audio.")
        return None
    return mfccs_processed

# Function to record audio from microphone
def record_audio(duration=5, sample_rate=22050):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording stopped.")
    return recording.flatten(), sample_rate

# Load pre-trained model (Assuming the model is saved as 'emotion_model.pkl')
model = joblib.load('emotion_model.pkl')

# Record audio from microphone
audio, sample_rate = record_audio(duration=5)  # Record for 5 seconds

# Extract features
features = extract_features(audio, sample_rate)
if features is not None:
    features = np.array([features])  # Make it compatible for prediction
    predicted_emotion = model.predict(features)
    print("Predicted Emotion:", predicted_emotion[0])
else:
    print("Feature extraction failed.")
