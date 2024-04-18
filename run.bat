@echo off

rem Nom de l'environnement virtuel
set "venv_name=emobot"

rem Vérifier si l'environnement virtuel existe déjà
if not exist "%venv_name%" (
    echo Création de l'environnement virtuel...
    python -m venv "%venv_name%"
)

rem Activer l'environnement virtuel
call "%venv_name%\Scripts\activate"

rem Installation des dépendances
pip install -r requirements.txt

rem Exécution du script Python
python streamlit run a_app.py

rem Désactiver l'environnement virtuel
deactivate