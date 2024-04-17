import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        print(f"Exception: {e}")
        return None

def load_data(dataset_path):
    features, labels = [], []
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.lower().endswith('.wav'):
                    data = extract_features(file_path)
                    if data is not None:
                        features.append(data)
                        labels.append(folder)  
    return np.array(features), np.array(labels)

try:
    features_dcv, labels_dcv = load_data('Acted Emotional Speech Dynamic Database')
    features_darvess, labels_darvess = load_data('archive')
    features_tess, labels_tess = load_data('TESS Toronto emotional speech set data')

    if features_dcv.size > 0 and features_darvess.size > 0 and features_tess.size > 0:
        features = np.concatenate((features_dcv, features_darvess, features_tess))
        labels = np.concatenate((labels_dcv, labels_darvess, labels_tess))
    else:
        print("Some datasets are empty or have different dimensions, can't proceed with concatenation.")
        labels = np.array([])  

except Exception as e:
    print(f"Failed to load data: {e}")
    features = np.array([])
    labels = np.array([])

if labels.size > 0:
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(256, input_shape=(40,), activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(len(np.unique(labels_encoded)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_accuracy}")

    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    model.save('emotion_recognition_model.keras')
    print("Model saved successfully!")

else:
    print("No data available to train the model. Check dataset paths and file permissions.")
