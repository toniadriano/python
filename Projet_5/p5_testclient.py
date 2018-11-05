# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 20:16:04 2018

@author: Toni
"""

import warnings
import random as rd
import pandas as pd

from sklearn.externals import joblib

# Pour ne pas avoir les warnings lors de la compilation
warnings.filterwarnings("ignore")

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p5\\'
_DOSSIERPKL = 'C:\\Users\\Toni\\python\\python\\Projet_5\\pkl'
_FICHIERDATA = _DOSSIER + 'dataset_p5.csv'

def creation_factures(data, nb_factures):
    """
    Fonction pour créer les factures mystères
    """

    # Création d'un dataframe vide avec les bonnes colonnes
    df_cli = pd.DataFrame(columns=data.columns)
    del df_cli['labels']

    for i in range(0, nb_factures):

        print('\nFacture n°', i)

        # 1 ligne = 1 facture
        row_c = i

        # Tirage au sort des valeurs de la facture
        df_cli.loc[row_c, 'nb_factures'] = 1
        df_cli.loc[row_c, 'somme_total'] = rd.randint(2, 500)
        df_cli.loc[row_c, 'nb_categorie_total'] = rd.randint(1, 500)
        df_cli.loc[row_c, 'nb_article_total'] = rd.randint(1, 100)
        df_cli.loc[row_c, 'day_of_week'] = rd.randint(0, 6)
        df_cli.loc[row_c, 'interval_jour_achat_n1'] = rd.randint(0, 12)
        df_cli.loc[row_c, 'interval_heure_achat_n1'] = rd.randint(0, 4)
        df_cli.loc[row_c, 'interval_moyenne_horaire'] = rd.randint(0, 4)
        df_cli.loc[row_c, 'valeur_facture_1'] = df_cli.loc[row_c, 'somme_total']

        # Déduction des autres
        df_cli.loc[row_c, 'mean_nb_article_facture'] = (
            df_cli.loc[row_c, 'nb_article_total']/df_cli.loc[row_c, 'nb_factures'])

        df_cli.loc[row_c, 'mean_somme_par_facture'] = (
            df_cli.loc[row_c, 'somme_total']/df_cli.loc[row_c, 'nb_factures'])

        df_cli.loc[row_c, 'mean_nb_categorie_facture'] = (
            df_cli.loc[row_c, 'nb_categorie_total']/df_cli.loc[row_c, 'nb_factures'])

        df_cli.loc[row_c, 'mean_somme_par_article'] = (
            df_cli.loc[row_c, 'somme_total']/df_cli.loc[row_c, 'nb_article_total'])

        df_cli.loc[row_c, 'ecart_moy_2_achats'] = 365
        df_cli.loc[row_c, 'ecart_min_2_achats'] = 365
        df_cli.loc[row_c, 'ecart_max_2_achats'] = 365

        # Affichage de confirmation
        for j in df_cli:
            print(j, "\t", df_cli.loc[row_c, j])

    return df_cli

def prediction_classe(df_cli, rfc):
    """
    Fonction pour prédire la classe du client pour chaque facture
    """

    for i in range(0, len(df_cli)):
        resultat = rfc.predict(df_cli.loc[i].reshape(1, -1))
        print('\nFacture n°', i, '\nLe client appartient à la classe n°', resultat)

def main(nb_factures):
    """
    Fonction principale
    """

    # Lecture du dataset
    data = pd.read_csv(_FICHIERDATA, error_bad_lines=False)
    del data['Unnamed: 0']

    # Nom du classifieur
    name = 'RandomForestClassifier'

    # Localisation de du fichier du fit sauvegardé
    fichier = _DOSSIERPKL + "\\" + name + ".pkl"

    # On va chercher le dump
    rfc = joblib.load(fichier)

    # Création aléatoire de factures
    df_cli = creation_factures(data, nb_factures)

    # Déduction du label et affichage
    prediction_classe(df_cli, rfc)
