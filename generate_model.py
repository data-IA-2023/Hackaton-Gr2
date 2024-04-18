import pandas as pd
import numpy as np
import os
import sys
import time
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pickle

# Paths for datasets, can be found here : https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition/input
Ravdess = "ressource_audio/audio_speech_actors_01-24/"
Crema = "ressource_audio/Crema/"
Tess = "ressource_audio/TESS Toronto emotional speech set data/"
Savee = "ressource_audio/Savee/"

'''============================ DataFrame avec Ravdess ============================'''

# Liste des répertoires dans le répertoire racine RAVDESS
ravdess_directory_list = os.listdir(Ravdess)

# Initialisation des listes pour stocker les émotions et les chemins des fichiers
file_emotion = []
file_path = []

# Parcours des répertoires dans RAVDESS
for dir in ravdess_directory_list:
    # Liste des fichiers dans chaque répertoire
    actor = os.listdir(Ravdess + dir)
    
    # Parcours des fichiers dans chaque répertoire
    for file in actor:
        # Extraction du code d'émotion du nom de fichier
        part = file.split('.')[0]
        part = part.split('-')
        file_emotion.append(int(part[2]))  # Ajout du code d'émotion à la liste
        file_path.append(Ravdess + dir + '/' + file)  # Construction du chemin du fichier et ajout à la liste

# Création des DataFrames pandas pour les émotions et les chemins des fichiers
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])

# Fusion des DataFrames en un seul
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# Remplacement des codes numériques d'émotion par leurs noms correspondants
Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)

'''============================ DataFrame avec Crema ============================'''

# Liste des fichiers dans le répertoire Crema
crema_directory_list = os.listdir(Crema)

# Initialisation des listes pour stocker les émotions et les chemins des fichiers
file_emotion = []
file_path = []

# Parcours des fichiers dans le répertoire Crema
for file in crema_directory_list:
    # Construction du chemin complet du fichier et ajout à la liste
    file_path.append(Crema + file)
    
    # Extraction de la partie contenant l'émotion du nom de fichier
    part = file.split('_')
    
    # Mapping des codes d'émotion aux émotions correspondantes
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')  # Si l'émotion n'est pas reconnue, elle est marquée comme 'Unknown'

# Création des DataFrames pandas pour les émotions et les chemins des fichiers
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])

# Fusion des DataFrames en un seul
Crema_df = pd.concat([emotion_df, path_df], axis=1)

'''============================ DataFrame avec Tess ============================'''

# Liste des répertoires dans le répertoire Tess
tess_directory_list = os.listdir(Tess)

# Initialisation des listes pour stocker les émotions et les chemins des fichiers
file_emotion = []
file_path = []

# Parcours des répertoires dans le répertoire Tess
for dir in tess_directory_list:
    # Liste des fichiers dans chaque sous-répertoire
    directories = os.listdir(Tess + dir)
    
    # Parcours des fichiers dans chaque sous-répertoire
    for file in directories:
        # Extraction du nom de fichier sans extension
        part = file.split('.')[0]
        
        # Extraction de l'émotion à partir du nom de fichier
        part = part.split('_')[2]
        
        # Mapping des émotions à partir des codes
        if part == 'ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        
        # Construction du chemin complet du fichier et ajout à la liste
        file_path.append(Tess + dir + '/' + file)

# Création des DataFrames pandas pour les émotions et les chemins des fichiers
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])

# Fusion des DataFrames en un seul
Tess_df = pd.concat([emotion_df, path_df], axis=1)

'''============================ DataFrame avec Savee ============================'''

# Liste des fichiers dans le répertoire Savee
savee_directory_list = os.listdir(Savee)

# Initialisation# Utilisez cette fonction avec votre propre fichier audio des listes pour stocker les émotions et les chemins des fichiers
file_emotion = []
file_path = []

# Parcours des fichiers dans le répertoire Savee
for file in savee_directory_list:
    # Construction du chemin complet du fichier et ajout à la liste
    file_path.append(Savee + file)
    
    # Extraction de l'élément correspondant à l'émotion du nom de fichier
    part = file.split('_')[1]
    
    # Traitement de l'élément pour obtenir l'émotion
    ele = part[:-6]
    
    # Mapping des éléments d'émotion aux émotions correspondantes
    if ele == 'a':
        file_emotion.append('angry')
    elif ele == 'd':
        file_emotion.append('disgust')
    elif ele == 'f':
        file_emotion.append('fear')
    elif ele == 'h':
        file_emotion.append('happy')
    elif ele == 'n':
        file_emotion.append('neutral')
    elif ele == 'sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')

# Création des DataFrames pandas pour les émotions et les chemins des fichiers
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])

# Fusion des DataFrames en un seul
Savee_df = pd.concat([emotion_df, path_df], axis=1)

'''============================ Concat ============================'''

data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)

'''============================ Data Augmentation ============================'''

# Fonction pour ajouter du bruit au signal audio
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

# Exemple d'utilisation des fonctions sur un échantillon de données audio
path = np.array(data_path.Path)[1]  # Sélection d'un chemin de fichier audio depuis un tableau de chemins
data, sample_rate = librosa.load(path)  # Chargement du fichier audio et récupération du taux d'échantillonnage

'''============================ Data Extraction ============================'''

# Fonction pour extraire les caractéristiques audio
def extract_features(data):
    # ZCR (Taux de franchissement de zéro)
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # Empilement horizontal

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # Empilement horizontal

    # MFCC (Cepstraux de fréquence en coefficients Mel)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # Empilement horizontal

    # Valeur quadratique moyenne
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # Empilement horizontal

    # MelSpectogramme
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # Empilement horizontal

    return result

# Fonction pour obtenir les caractéristiques avec augmentation de données
def get_features(path):
    # Chargement du fichier audio avec une durée spécifique et un décalage
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # Extraction des caractéristiques sans augmentation
    res1 = extract_features(data)
    result = np.array(res1)
    
    # Extraction des caractéristiques avec ajout de bruit
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # Empilement vertical
    
    # Extraction des caractéristiques avec étirement et changement de hauteur tonale
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))  # Empilement vertical
    
    return result

# Initialisation des listes pour les caractéristiques (X) et les étiquettes (Y)
X, Y = [], []

# Parcours des chemins de fichiers et des émotions correspondantes
for path, emotion in zip(data_path.Path, data_path.Emotions):
    # Extraction des caractéristiques pour chaque fichier avec augmentation
    feature = get_features(path)
    
    # Ajout des caractéristiques et des émotions à X et Y
    for ele in feature:
        X.append(ele)
        # Ajout de l'émotion trois fois en raison de trois techniques d'augmentation par fichier audio
        Y.append(emotion)

# Création d'un DataFrame pandas pour les caractéristiques
Features = pd.DataFrame(X)
Features['labels'] = Y

'''============================ Data Preparation ============================'''

# Séparation des caractéristiques (X) et des étiquettes (Y)
X = Features.iloc[:, :-1].values
Y = Features['labels'].values

# Encodage one-hot des étiquettes
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

output = open('pickle_save/Departure_encoder.pkl', 'wb')
pickle.dump(encoder, output)
output.close()

# Division des données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)

# Mise à l'échelle des caractéristiques avec StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Ajout d'une dimension pour correspondre à l'entrée attendue du modèle (3D pour CNN)
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

# Affichage des dimensions des ensembles d'entraînement et de test
x_train.shape, y_train.shape, x_test.shape, y_test.shape

'''============================ Modeling ============================'''

# Création du modèle séquentiel
model = Sequential()

# Couche de convolution 1D avec 256 filtres, taille de noyau 5 et fonction d'activation ReLU
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))

# Couche de max pooling 1D pour réduire la taille spatiale des caractéristiques
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

# Autres couches de convolution et de max pooling avec des paramètres similaires
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(Dropout(0.2))  # Couche de régularisation pour réduire le surapprentissage
model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

# Couche d'aplanissement pour préparer les données à l'entrée d'une couche dense
model.add(Flatten())

# Couche dense avec 32 neurones et fonction d'activation ReLU
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))  # Couche de régularisation

# Couche de sortie avec 8 neurones pour les 8 classes d'émotions et activation softmax pour la classification
model.add(Dense(units=8, activation='softmax'))

# Compilation du modèle avec l'optimiseur Adam, la perte de catégorie croisée catégorique et la métrique d'exactitude
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Affichage du résumé du modèle
model.summary()

# Réduction du taux d'apprentissage si la perte ne diminue pas après un certain nombre d'époques
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)

# Entraînement du modèle avec les données d'entraînement et validation sur les données de test
history = model.fit(x_train, y_train, batch_size=64, epochs=122, validation_data=(x_test, y_test), callbacks=[rlrp])

# Évaluation de la précision du modèle sur les données de test
print("Accuracy of our model on test data : ", model.evaluate(x_test, y_test)[1] * 100, "%")

# Tracé des courbes d'apprentissage (perte et exactitude) au fil des époques
epochs = [i for i in range(122)]
fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(20, 6)
ax[0].plot(epochs, train_loss, label='Training Loss')
ax[0].plot(epochs, test_loss, label='Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs, train_acc, label='Training Accuracy')
ax[1].plot(epochs, test_acc, label='Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()

# Prédiction sur les données de test
pred_test = model.predict(x_test)

# Conversion des prédictions en étiquettes
y_pred = encoder.inverse_transform(pred_test)
y_test = encoder.inverse_transform(y_test)

# Création d'un DataFrame pour les prédictions et les étiquettes réelles
df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

# Calcul et affichage de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
cm = pd.DataFrame(cm, index=[i for i in encoder.categories_], columns=[i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()

# Affichage du rapport de classification
print(classification_report(y_test, y_pred))

'''============================ Saving Model ============================'''

model_name = 'model_final.h5'
save_dir = "model_save"

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

# Check if the file already exists
if os.path.exists(model_path):
    # If it does, find a new filename
    base_name, ext = os.path.splitext(model_name)
    i = 1
    while os.path.exists(os.path.join(save_dir, f"{base_name} ({i}){ext}")):
        i += 1
    model_name = f"{base_name} ({i}){ext}"
    model_path = os.path.join(save_dir, model_name)

model.save(model_path)
print('Saved trained model at %s ' % model_path)

'''============================ END ============================'''

print("If you reached this message, everything is probably okay")