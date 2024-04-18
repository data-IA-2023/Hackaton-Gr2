#!/bin/bash

# Nom de l'environnement virtuel
venv_name="emobot"

# Création de l'environnement virtuel s'il n'existe pas
if [ ! -d "$venv_name" ]; then
    echo "Création de l'environnement virtuel..."
    python3.12 -m venv "$venv_name"
fi

# Activation de l'environnement virtuel
source "$venv_name/bin/activate"

# Installation des dépendances
pip install -r requirements.txt

# Exécution du script Python
python streamlit run a_app.py

# Désactivation de l'environnement virtuel
deactivate