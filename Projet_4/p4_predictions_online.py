# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:25:20 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import json
import pandas as pd
from flask import Flask
from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")

# Lieu où se trouve le fichier
_DOSSIERWEB = '/home/pfroide/mysite/p4/'

# Création de la route d'entrée de l'API
app = Flask(__name__)
@app.route('/prediction/<chaine>')

def web_predict(chaine):
    """
    Fonction principale de prédiction
    """

    # On récupère la chaine d'entrée et on la splite dans un array
    listing = chaine.split("_")

    # Le classificateur que j'ai choisi
    clf = 'SGDRegressor'

    # Le nom de la compagnie est dans la dernière case de l'array d'entrée
    compagnie = listing[5]

    # Création du nom du fichier de data utilisé
    _FICHIERDATA = _DOSSIERWEB + 'dataset_p4_' + compagnie + '.csv'

    # Récupération des dataset
    data = pd.read_csv(_FICHIERDATA, error_bad_lines=False, low_memory=False)
    del data['Unnamed: 0']

    # Mise en conformité des mois (à cause de certains floats)
    data['MONTH'] = data['MONTH'].astype('int', copy=False)

    # Suppression des deux colonnes qui ne nous servent pas
    del data['UNIQUE_CARRIER']
    del data['ARR_DELAY']

    # Transposition en 0 et 1 des valeurs non-numériques
    liste_criteres = ['ORIGIN',
                      'DEP_TIME_BLK',
                      'DAY_OF_WEEK',
                      'DEST',
                      'MONTH']

    # One-Hot encoding
    data = pd.get_dummies(data=data, columns=liste_criteres)

    # Récupération du fichier joblib de sauvegarde des train
    fichier = _DOSSIERWEB + clf + "_" + compagnie + ".pkl"
    clf = joblib.load(fichier)

    # Appel de la fonction de prédiction
    retard = predict(data, clf, listing)

    # Résultat de la prédiction, transformation en json
    if retard == -100:
        # Transformation en dictionnaire
        dico = {"Retard" : "Cette liaison n'existe pas"}

    else:
        # Transformation en dictionnaire
        dico = {"Retard" : round(retard[0], 0)}

    # Transformation en format JSON
    json_dico = json.dumps(dico, indent=4, separators=(',', ': '))

    # On renvoie le résultat en format JSON
    return json_dico

def predict(data, clf, listing):

    # Création du dataframe vide pour avoir le nom des colonnes
    colonnes = data.columns
    data_vide = pd.DataFrame([], columns = colonnes)

    # Création du dataframe pour la prédiction
    colonnes = ['ORIGIN', 'DEST', 'DEP_TIME_BLK', 'DAY_OF_WEEK', 'MONTH']
    res = pd.DataFrame([[listing[0], listing[1], listing[2], listing[3], listing[4]]], columns = colonnes)
    data_res = pd.get_dummies(data=res)

    # Concaténation
    res2 = pd.concat([data_vide, data_res])
    res2.fillna(0, inplace=True)

    # Pour déterminer si la liaison existe, on vérifie la longueur des deux datasets
    if len(data.columns) == len(res2.columns):
        prediction_retard = clf.predict(res2)
    else:
        prediction_retard = -100

    # On renvoie la valeur trouvée
    return prediction_retard
