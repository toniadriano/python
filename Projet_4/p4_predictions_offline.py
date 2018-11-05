# -*- coding: utf-8 -*-
"""
Created on Sun Apr 1 21:12:42 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import pandas as pd
from sklearn.externals import joblib
    
import warnings
warnings.filterwarnings("ignore")

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p4\\'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_4\\pkl'
_FICHIERDATA = _DOSSIER + 'dataset_p4.csv'

listing = ['ANC', 'SEA', '1500-1559', '6', '8', 'AS']

def main(clf, listing):
    """
    Fonction main
    """

    # Récupération des dataset
    data = pd.read_csv(_FICHIERDATA, error_bad_lines=False, low_memory=False)
    del data['Unnamed: 0']   

    data['MONTH'] = data['MONTH'].astype('int', copy=False)
    
    compagnie = listing[5]

    data = data[data['UNIQUE_CARRIER']==compagnie]
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

    #
    fichier = _DOSSIERTRAVAIL + "\\" + clf + "_" + compagnie + ".pkl"
    clf = joblib.load(fichier)

    retard = predict(data, clf, listing)

    if retard == -1:
        print("Cette liaision n'existe pas")
    else:
        #print("Retard prédit : ", round(retard, 3))
        print("Retard prédit : ", retard)

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
    
    if len(data.columns) == len(res2.columns):
        prediction_retard = clf.predict(res2)
    else:
        prediction_retard = -1
        
    return prediction_retard
