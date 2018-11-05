# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 20:33:54 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, cm as cm

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p4\\'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_4\\images'

def correlation_matrix(data):
    """
        Fonction qui permet de créer une visualisation du lien entre les
        variables 2 à 2
    """

    # Calcule de la matrice
    corr = data.corr()
    cmap = cm.get_cmap('jet', 30)

    # Taille de la figure
    plt.figure(figsize=(15, 15))
    # Création du type d'image
    cax = plt.imshow(data.corr(), interpolation="nearest", cmap=cmap)
    plt.grid(True)

    # Libellés sur les axes
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=10)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=10)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    plt.colorbar(cax, ticks=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    plt.tight_layout()
    plt.savefig(_DOSSIERTRAVAIL + '\\matrix', dpi=100)
    plt.show()

def histogramme(data, colon, limitemin, limitemax):
    """
        Note : La première colonne et la dernière ont un " caché
    """

    fichier_save = _DOSSIERTRAVAIL + '\\' + 'histogram_' + colon

    #steps = (max(data[colon])-min(data[colon]))/100
    #bin_values = np.arange(start=min(data[colon]), stop=max(data[colon]), step=steps)
    plt.figure(figsize=(10, 6))
    plt.xlabel('Valeurs')
    plt.ylabel('Décompte')
    titre = 'Histogramme ' + colon
    plt.title(titre)
    plt.xlim(limitemin, limitemax)
    # plt.hist(data[colon], bins=bin_values)
    # Test sans les valeurs NaN
    classes = np.linspace(-100, 100, 200)

    # Ligne rouge verticale
    plt.plot([0.0, 0], [0, 160000], 'r-', lw=2)

    # Données de l'histogramme
    plt.hist(data[colon][np.isfinite(data[colon])], bins=classes)
    plt.tight_layout()
    plt.savefig(fichier_save, dpi=100)

def get_stats(param):
    """
    TBD
    """
    return {'min':param.min(),
            'max':param.max(),
            'count': param.count(),
            'mean':param.mean()
           }

def graphique_par_donnee(data, classifieur):
    """
    TBD
    """

    axe_y = 'ARR_DELAY'

    # nom du fichier de sauvegarde
    fichier_save = _DOSSIERTRAVAIL + '\\' + classifieur

    # On range les données en faisaint la moyenne
    dep = data[[axe_y, classifieur]].groupby(classifieur, as_index=True).mean()

    # Tipe d'affichage et taille
    axe = dep.plot(kind="bar", figsize=(len(dep)/2, 6))

    # On fait les labels pour les afficher
    labels = ["%.2f" % i for i in dep[axe_y]]
    rects = axe.patches

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        width = rect.get_width()

        # Différence entre chiffres négatifs et positifs
        if "-" not in label:
            axe.text(rect.get_x() + width / 2, height + 0.1, label, ha='center', va='bottom')
        else:
            axe.text(rect.get_x() + width / 2, height - 0.6, label, ha='center', va='bottom')

    # Titres
    axe.set_xlabel(classifieur, fontsize=10)
    axe.set_ylabel(axe_y, fontsize=10)
    titre = "Retards pour " + classifieur
    axe.set_title(titre, fontsize=16)

    # on supprime la légende
    axe.legend().set_visible(False)

    # Sauvegarde de la figure
    fig = axe.get_figure()
    plt.tight_layout()
    fig.savefig(fichier_save, dpi=100)

def classification_retards(data):
    """
    TBD
    """
    
    # Classification des retards
    for dataset in data:
        data.loc[data['ARR_DELAY'] <= 15, 'CLASSE_DELAY'] = "Leger"
        data.loc[data['ARR_DELAY'] >= 15, 'CLASSE_DELAY'] = "Moyen"
        data.loc[data['ARR_DELAY'] >= 45, 'CLASSE_DELAY'] = "Important"
        data.loc[data['ARR_DELAY'] < 0, 'CLASSE_DELAY'] = "En avance"

    # Affichage de la classification des retards
    f, ax = plt.subplots(1, 2, figsize=(20, 8))
    data['CLASSE_DELAY'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title('CLASSE_DELAY')
    ax[0].set_ylabel('')
    sns.countplot('CLASSE_DELAY',order=data['CLASSE_DELAY'].value_counts().index, data=data, ax=ax[1])
    ax[1].set_title('Status')
    plt.tight_layout()
    plt.savefig(_DOSSIERTRAVAIL + '\\' + 'classification_retards', dpi=100)
    plt.show()
    
    del data['CLASSE_DELAY']
    
def main():
    """
    Fonction main
    """

    # Récupération des dataset
    # Pour toute l'année
    data = pd.DataFrame({'A' : []})
    for i in range(1, 13):
        if i < 10:
            fichier = str('2016_0' + str(i) + '.csv')
        else:
            fichier = str('2016_' + str(i) + '.csv')

        datatemp = pd.read_csv(_DOSSIER + fichier, error_bad_lines=False, low_memory=False)

        # Suppresion des données fausses
        datatemp = datatemp[datatemp['MONTH'] == i]

        data = pd.concat([data, datatemp])

    # Conversion en int de valeurs qui ne le seraient pas.
    data['DAY_OF_WEEK'] = data['DAY_OF_WEEK'].astype('int', copy=False)
    data['DAY_OF_MONTH'] = data['DAY_OF_MONTH'].astype('int', copy=False)
    data['ORIGIN_AIRPORT_ID'] = data['ORIGIN_AIRPORT_ID'].astype('float', copy=False)
    data['ORIGIN_AIRPORT_ID'] = data['ORIGIN_AIRPORT_ID'].astype('int', copy=False)

    # Liste des critères à conserver pour les stats
    liste_criteres = []
    liste_criteres = ['FL_DATE', 'AIRLINE_ID', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID']
    liste_criteres.extend(['DEP_TIME', 'DEP_DELAY', 'ARR_TIME', 'ARR_DELAY'])
    liste_criteres.extend(['DISTANCE', 'AIR_TIME', 'LATE_AIRCRAFT_DELAY'])
    liste_criteres.extend(['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY'])
    liste_criteres.extend(['DAY_OF_WEEK', 'MONTH', 'ORIGIN', 'DEST'])
    liste_criteres.extend(['DAY_OF_MONTH', 'UNIQUE_CARRIER', 'DEP_TIME_BLK', 'ARR_TIME_BLK'])
    liste_criteres.extend(['ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'DISTANCE_GROUP'])

    # Suppression des colonnes non-selectionnées
    for colon in data:
        if colon not in liste_criteres:
            del data[colon]

    # Suppression des lignes qui seraient en double
    data = data.drop_duplicates(keep='first')

    # Classification des retards
    # Affichage de la classification des retards
    classification_retards(data)

    # Données manquantes
    print("Données manquantes")
    fichier_save = _DOSSIERTRAVAIL + '\\' + 'missing_data.csv'
    missing_data = data.isnull().sum(axis=0).reset_index()
    missing_data.columns = ['column_name', 'missing_count']
    missing_data['filling_factor'] = (data.shape[0]-missing_data['missing_count'])/data.shape[0]*100
    print(missing_data.sort_values('filling_factor').reset_index(drop=True))
    missing_data.sort_values('filling_factor').reset_index(drop=True).to_csv(fichier_save)

    # Transposition du dataframe de données pour l'analyse univariée
    fichier_save = _DOSSIERTRAVAIL + '\\' + 'transposition.csv'
    data_transpose = data.describe().reset_index().transpose()
    print(data_transpose)
    data_transpose.to_csv(fichier_save)

    # Affichage de la matrice de corrélation
    correlation_matrix(data)

    # Données pour affichage
    liste_affichage = ['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'MONTH']
    liste_affichage.extend(['AIRLINE_ID', 'DAY_OF_WEEK', 'DISTANCE_GROUP'])
    liste_affichage.extend(['UNIQUE_CARRIER', 'DEP_TIME_BLK', 'ARR_TIME_BLK'])

    # Affichage de graphiques
    for colon in liste_affichage:
        graphique_par_donnee(data, colon)

    # Deuxième partie de l'affichage précédent
    for colon in liste_affichage:
        print("Nb of", colon, " : ", len(data[colon].unique()))

    # Création des histogrammes
    histogramme(data, 'ARR_DELAY', -60, 60)
    histogramme(data, 'DEP_DELAY', -60, 60)

    # Il faut que les différents délais soient inférieurs à 45 min
    retard_max = 45
    mask = (data["CARRIER_DELAY"] <= retard_max) | (data["ARR_DELAY"] <= 15)
    data = data.loc[mask]

    mask = (data["LATE_AIRCRAFT_DELAY"] <= retard_max) | (data["ARR_DELAY"] <= 15)
    data = data.loc[mask]

    mask = (data["NAS_DELAY"] <= retard_max) | (data["ARR_DELAY"] <= 15)
    data = data.loc[mask]

    mask = (data["SECURITY_DELAY"] <= retard_max) | (data["ARR_DELAY"] <= 15)
    data = data.loc[mask]

    mask = (data["WEATHER_DELAY"] <= retard_max) | (data["ARR_DELAY"] <= 15)
    data = data.loc[mask]

    mask = data["ARR_DELAY"] >= -retard_max
    data = data.loc[mask]

    mask = data["ARR_DELAY"] <= retard_max
    data = data.loc[mask]
    
    # Préparation de l'exportation des données
    liste_a_supprimer = ['AIRLINE_ID', 'AIR_TIME', 'ARR_TIME', 'CARRIER_DELAY']
    liste_a_supprimer.extend(['DAY_OF_MONTH', 'DEP_TIME', 'DISTANCE', 'DEP_DELAY'])
    liste_a_supprimer.extend(['NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY'])
    liste_a_supprimer.extend(['ORIGIN_CITY_NAME', 'ORIGIN_AIRPORT_ID', 'ARR_TIME_BLK'])
    liste_a_supprimer.extend(['FL_DATE', 'DEST_CITY_NAME', 'DEST_AIRPORT_ID'])
    liste_a_supprimer.extend(['DISTANCE_GROUP', 'LATE_AIRCRAFT_DELAY'])

    # Suppression
    for donnee in liste_a_supprimer:
        del data[donnee]

    # Affichage de la classification des retards
    classification_retards(data)

    #     
    df_copy = data.copy()
    
    #
    for cp in data['UNIQUE_CARRIER'].unique():
        # Export - On ne garde que les données pour 1 compagnie à la fois
        df_copy = data[data['UNIQUE_CARRIER'] == cp]
        df_copy.to_csv('C:\\Users\\Toni\\Desktop\\dataset_p4_' + cp + '.csv')
