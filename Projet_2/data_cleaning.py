"""
    Projet n°2.
    OpenFood
"""
#! /usr/bin/env python3
# coding: utf-8

# On importe les librairies dont on aura besoin pour ce tp
import pandas as pd
from matplotlib import pyplot as plt

# Valeur limite des Nan acceptée
# Il faut donc moins de _VALEUR_LIMITE_NAN valeurs "NaN" pour garder la colonne
_VALEUR_LIMITE_NAN = 200000

# Référence en y du plot
_ORDONNEE = "nutrition-score-fr_100g"

# Lieu où se trouve le FICHIER
_FICHIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\bdd.csv'

# que faire des données manquantes ?
# affichage avant/après traitements
# faire un modèle de prédiction du score nutrionnel (continue/classe)

def remplir_colonnes(data, nom_colonne, colonnes):
    """
        Fonction qui permets de selectionner des colonnes pour la
        base de données
    """

    # Log
    print("\nFct remplir_colonnes : Traitement de : %s" % nom_colonne)

    # test du type de la colonne. IL n'y a que les valeurs numériques qui
    # nous intéressent
    if data[nom_colonne].dtype == 'float':
        # si "100g" est trouvé dans le nom de la colonne
        if nom_colonne.find('100g') != -1 and nom_colonne.find('uk') == -1:
            colonnes.append(nom_colonne)
            print("Cette donnée est gardée")
        else:
            print("Cette donnée est exclue : pas de 100g")

    else:
        print("Cette donnée est exclue : pas un float")

def supprimer_colonnes(data, nom_colonne):
    """
        Fonction qui permets de supprimer des colonnes de la bdd
    """

    # Log
    print("\nFct supprimer_colonnes : Traitement de : %s" % nom_colonne)

    # nombre de valeurs "NaN"
    # .isnull().sum() = nombre par ligne
    # .isnull().sum().sum() = nombre total
    cpt_nan = data[nom_colonne].isnull().sum().sum()

    # S'il y a plus de valeur "Nan" que le chiffre défini, on vire la colonne
    if cpt_nan > (_VALEUR_LIMITE_NAN):
        # Suprresion de la colonne
        del data[nom_colonne]

        # Log
        print("Cette donnée est exclue : elle contient %.0f 'NaN' " % cpt_nan)
    else:
        # Log
        print("Cette donnée est gardée : elle contient %.0f 'NaN' " % cpt_nan)

def fct_missing_data(data):
    """
        Statistiques sur les données manquantes
    """
    # Compte les données manquantes par colonne
    missing_data = data.isnull().sum(axis=0).reset_index()

    # Change les noms des colonnes
    missing_data.columns = ['column_name', 'missing_count']

    # Crée une nouvelle colonne et fais le calcul en pourcentage des données
    # manquantes
    missing_data['fill_fact'] = (data.shape[0]-missing_data['missing_count']) / data.shape[0] * 100

    # Classe et affiche
    print(missing_data.sort_values('fill_fact', ascending=False).reset_index(drop=True))

    # Affichage de la valeur moyenne de données vides
    print("% fill_fact : ", missing_data['fill_fact'].mean())

def affichage_plot(data, nom_colonne):
    """
        Fonction qui permet d'afficher les nuages de points
    """

    #Log
    print("Fct affichage_plot : Affichage de la courbe\n")

    # Déliminations du visuel pour x
    xmax = max(data[nom_colonne])
    ymax = max(data[_ORDONNEE])

    # Déliminations du visuel pour y
    xmin = min(data[nom_colonne])
    ymin = min(data[_ORDONNEE])

    # création du nuage de point avec toujours la même ordonnée
    data.plot(kind="scatter", x=nom_colonne, y=_ORDONNEE)

    # Affichage
    plt.grid(True)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()

def main():
    """
        Note : La première colonne et la dernière ont un " caché
    """

    # Définition de la variable qui récupère le nom des colonnes
    colonnes = []

    # On charge la première ligne du dataset
    bdd_titres = pd.read_csv(_FICHIER,
                             nrows=1,
                             error_bad_lines=False,
                             engine='python',
                             sep=r'\t')

    # Fonction qui va choisir les colonnes à récupérer suivant les crières
    # définis
    for i in bdd_titres:
        remplir_colonnes(bdd_titres, i, colonnes)

    # Rajout manuel d'une colonne intéressante
    #colonnes.append("nutrition_grade_fr")

    # On charge le dataset sur les colonnes qui nous ont intéressés dans la
    # fonction du dessus
    data = pd.read_csv(_FICHIER,
                       usecols=colonnes,
                       error_bad_lines=False,
                       engine='python',
                       sep=r'\t')

    # Appel de le fonction qui va montrer les données manquantes
    fct_missing_data(data)

    # On supprime les lignes qui sont vides et n'ont que des "nan"
    # axis : {0 or ‘index’, 1 or ‘columns’},
    data = data.dropna(axis=1, how='all')
    data = data.dropna(how='all')

    # Suppression des colonnes qui ne remplissent pas les conditions posées
    for i in data:
        supprimer_colonnes(data, i)

    # Trouver le numéro de colonne qui nous sert d'ordonné dans l'affichage
    position_ordonne = data.columns.get_loc(_ORDONNEE)

    # Affichage des nuages de points avant traitement
    for i in data.columns.values[0:position_ordonne]:
        # Log
        print("Avant traitement")
        affichage_plot(data, i)

    # Log
    print("Fct traitement_data : \n")

    for nom_colonne in data.columns.values[0:position_ordonne]:
        # On garde les valeurs positives
        data = data[data[nom_colonne] >= 0]
        # On prends 98% de toutes les valeurs pour couper les grandes valeurs
        # farfelues
        data = data[data[nom_colonne] <= data[nom_colonne].quantile(0.98)]

    # Affichage des nuages de points après traitement
    for i in data.columns.values[0:position_ordonne]:
        # Log
        print("Après traitement")
        affichage_plot(data, i)

    print("\n\nLa base de données est nettoyée.\n\n")

    # Appel de le fonction qui va montrer les données manquantes
    fct_missing_data(data)

    # Pour rajouter la colonne suivante 
    data_complet = pd.read_csv(_FICHIER,
                       error_bad_lines=False,
                       engine='python',
                       sep=r'\t')

    # Rajout d'une colonne
    data['nutrition_grade_fr']=data_complet['nutrition_grade_fr']

    # export csv
    data.to_csv('C:\\Users\\Toni\\Desktop\\pas_synchro\\bdd_clean.csv')
