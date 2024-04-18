# Projet EmoBot !

---

## Table des matières

- [Aperçu](#aperçu)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Contribuer](#contribuer)
- [Licence](#licence)
- [Contact](#contact)

---

## Aperçu

Le projet EmoBot est un projet étudiant dont les conditions étaient de réaliser en 4 jours seulement une application permettant de discuter avec un chatbot qui prendrait
en compte les émotions, notre groupe à eu pour condition de devoir intéréagir avec le dit bot par fonction vocal et une obtenir une réponse également par audio.

L'application est réalisée sur Streamlit, elle embarque 5 points importants :
  - Une méthode de TTS (Text-to-speech)
  - Une méthode de STT (Speech-to-text)
  - Une méthode de ChatBot (Mistral)
  - Une méthode d'enregistrement audio (PyAudio)
  - Une méthode de prédiction des émotions dans la voix (Keras)

---

## Installation

L'application fonctionne sous :
  - Windows 10/11
  - Linux debian based (Ubuntu, Kubutu, PopOs!....)
  - Mac OS Ventura/Sonoma

Conditions requises :
  - Python 3.12
  - Application web nécéssitant un navigateur

---

## Utilisation

__Windows :__

1. Téléchargez l'archive en .zip
2. Dézippez l'archive dans un dossier
3. Pour faire fonctionner le chatbot vous devez posseder un compte huggingface : https://huggingface.co/join, il vous faudra ensuite créer un fichier ``"credentials.env"`` dans le répértoire "credentials" (lire le fichier ``"place_keys_here"``) pour plus d'informations.
4. Lancez en tant qu'administrateur le fichier ``run.bat``

__Linux / MacOS :__

1. Téléchargez l'archive en .zip
2. Dézippez l'archive dans un dossier
3. Pour faire fonctionner le chatbot vous devez posseder un compte huggingface : https://huggingface.co/join, il vous faudra ensuite créer un fichier ``"credentials.env"`` dans le répértoire "credentials" (lire le fichier ``"place_keys_here"``) pour plus d'informations.
4. Lancez en tant qu'administrateur le fichier ``run.sh``

---

## Contribuer

Les contributions sont ce qui rend la communauté open source un endroit incroyable pour apprendre, inspirer et créer.
Cependant il s'agit d'un projet étudiant, je suis ouvert aux critiques et aux conseils sur mon travail cependant je n'apporterai aucun support utilisateur et probablement aucune mise à jour, merci de votre compréhension.

---

## Licence

Libre de droit I guess

---

## Contact

Discord : laamh

---

Projet Réalisé par : Yahya Ahmed, Naoufel Boutet, Kaelig Barillet, Paolo Peraza Muñoz, Clément Nogues
