import numpy as np

def valeurs_plus_probables(probas):
    valeurs = []
    pourcentages = []
    for ligne in probas:
        max_index = np.argmax(ligne)
        valeur = ligne[max_index]
        pourcentage = valeur * 100
        valeurs.append(valeur)
        pourcentages.append(pourcentage)
    return valeurs, pourcentages