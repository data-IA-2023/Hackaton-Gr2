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
  - Application web nécessitant un navigateur

---

## Utilisation

Attention : Le projet doit être contenu dans un dossier ne possédant pas d'espaces, sinon tout casser.

__Windows :__

1. Téléchargez l'archive en .zip : [Ici](https://github.com/data-IA-2023/Hackaton-Gr2/releases/)
2. Dézippez l'archive dans un dossier
3. Pour faire fonctionner le chatbot vous devez posseder un compte huggingface : https://huggingface.co/join, il vous faudra ensuite créer un fichier ``"credentials.env"`` dans le répértoire "credentials" (lire le fichier ``"place_keys_here"``) pour plus d'informations.
4. Pour que le STT (Speech to text) fonctionne convenablement il faut télécharger son modèle et le placer à la racine avec le nom ``model_stt``, vous pouvez trouver des modèles pré-entraînés dans plusieurs langues ici : https://alphacephei.com/vosk/models
5. Lancez en tant qu'administrateur le fichier ``run.bat``

__Linux / MacOS :__

1. Téléchargez l'archive en .zip : [Ici](https://github.com/data-IA-2023/Hackaton-Gr2/releases/)
2. Dézippez l'archive dans un dossier
3. Pour faire fonctionner le chatbot vous devez posseder un compte huggingface : https://huggingface.co/join, il vous faudra ensuite créer un fichier ``"credentials.env"`` dans le répértoire "credentials" (lire le fichier ``"place_keys_here"``) pour plus d'informations.
4. Pour que le STT (Speech to text) fonctionne convenablement il faut télécharger son modèle et le placer à la racine avec le nom ``model_stt``, vous pouvez trouver des modèles pré-entraînés dans plusieurs langues ici : https://alphacephei.com/vosk/models
5. Lancez en tant qu'administrateur le fichier ``run.sh``

### Exemple de ce à quoi doit ressembler votre répertoire (screenshot Windows 11)
![<Exemple>](https://media.discordapp.net/attachments/1223209671515439174/1230657115727597599/image.png?ex=66341dc0&is=6621a8c0&hm=a184c888635c3b0c54f1edc7ef77965d7111e5a0004fc0ff8caad7d5f1a600c8&=&format=webp&quality=lossless&width=809&height=302)

---

## Contribuer

Les contributions sont ce qui rend la communauté open source un endroit incroyable pour apprendre, inspirer et créer.
Cependant il s'agit d'un projet étudiant, je suis ouvert aux critiques et aux conseils sur mon travail cependant je n'apporterai aucun support utilisateur et probablement aucune mise à jour, merci de votre compréhension.

---

## Licence

License DWTFYW :
Do what the fuck you want

---

## Contact

Discord : laamh

---

Projet Réalisé par : Yahya Darwish, Naoufel Boutet, Kaelig Barillet, Paolo Peraza Muñoz, Clément Nogues
