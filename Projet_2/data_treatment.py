"""
    Projet n°2.
    OpenFood
"""
#! /usr/bin/env python3
# coding: utf-8

# On importe les librairies dont on aura besoin pour ce tp
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt, cm as cm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import linear_model

# Référence en y du plot
_ORDONNEE = "nutrition-score-fr_100g"

# Lieu où se trouve le FICHIER
_FICHIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\bdd_clean.csv'

# Lieu de sauvegarde
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_2\\images'

def definir_importance(data):
    """
        Fonction qui permet d'afficher un graphique
        où l'on verra l'importance relative de chaque élément dans le calcul
        final
    """

    # Log
    print("Fct definir_importance : \n")

    # Trouver le numéro de colonne qui nous sert d'ordonné dans l'affichage
    position_ordonne = data.columns.get_loc(_ORDONNEE)

    # Isolate Data, class labels and column values
    #X = data.iloc[:,0:15]       # colonne 0 à 15
    xdata = data.iloc[:, 0:position_ordonne]    # toutes les colonnes sauf la dernière
    ydata = data.iloc[:, position_ordonne]      # dernière colonne

    # names = data.columns.values[0:15]                 # avec limite
    names = data.columns.values[0:position_ordonne]     # sans limite

    # Build the model
    rfc = RandomForestClassifier(n_estimators=100, criterion='gini', verbose=1)

    # Fit the model
    rfc.fit(xdata, ydata)

    # Print the results
    #print("Features sorted by their score:")
    #print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), names), reverse = True))

    # Isolate feature importances
    importance = rfc.feature_importances_

    # Sort the feature importances
    sorted_importances = np.argsort(importance)

    # Insert padding
    padding = np.arange(len(names)) + 0.5

    # Customize the plot
    plt.yticks(padding, names[sorted_importances])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")

    # Plot the data
    plt.barh(padding, importance[sorted_importances], align='center')

    # Show the plot
    plt.show()

def regression_lineaire(data, colon1, colon2):
    """
        Fonction pour le calcul des régressions linéaires
    """

    ##fichier_save = _DOSSIERTRAVAIL + '\\' + 'regression_' + colon1 + '_' + colon2

    # Calcul de la droite optimale
    regr = linear_model.LinearRegression()
    regr.fit(data[colon1].values.reshape(-1, 1), data[colon2].values.reshape(-1, 1))
    score = np.corrcoef(data[colon1], data[colon2])[1, 0]

    # Affichage de la variances : On doit être le plus proche possible de 1
    if score > 0.5:
        print('Régression sur les deux données :', colon1, "et", colon2)
        print('Score : %.2f' % score)

    # Affichage de la droite optimale)
    # plt.plot([0, 200], [regr.intercept_, regr.intercept_ + 200*regr.coef_],
    # linewidth=2.0, label=regr.coef_)
    # plt.plot(data[colon1], data[colon2],'ro', markersize=1)
    # plt.legend()
    # plt.savefig(fichier_save, dpi=100)
    # plt.show()

def correlation_matrix(data):
    """
        Fonction qui permet de créer une visualisation du lien entre les
        variables 2 à 2
    """
    fichier_save = _DOSSIERTRAVAIL + '\\' + 'cor_matrix_'
    
    # Calcule de la matrice
    corr = data.corr()
    cmap = cm.get_cmap('jet', 30)

    # Taille de la figure
    plt.figure(figsize=(10, 10))
    # Création du type d'image
    cax = plt.imshow(data.corr(), interpolation="nearest", cmap=cmap)
    plt.grid(True)

    # Libellés sur les axes
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=15)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=15)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    plt.colorbar(cax, ticks=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    plt.savefig(fichier_save, bbox_inches='tight')
    plt.show()

def histogramme(data, colon):
    """
        Note : La première colonne et la dernière ont un " caché
    """

    fichier_save = _DOSSIERTRAVAIL + '\\' + 'histogram_' + colon

    steps = (max(data[colon])-min(data[colon]))/100
    bin_values = np.arange(start=min(data[colon]), stop=max(data[colon]), step=steps)
    plt.figure(figsize=(10, 6))
    plt.xlabel('Valeurs')
    plt.ylabel('Décompte')
    titre = 'Histogramme ' + colon
    plt.title(titre)
    plt.hist(data[colon], bins=bin_values)
    plt.savefig(fichier_save, dpi=100)

def sanite(df, dico, nb_sain):
    """
        TBD
    """
        # Rajout d'une colonne qui est True/False si les aliments sont sains
    df['Aliment Sain'] = df['Qualite'].map(dico)

    # Visualisation via un camembert
    data_pie_1 = df['Aliment Sain'].where(df['Aliment Sain']).count()
    data_pie_2 = df['Aliment Sain'].where(df['Aliment Sain'] == False).count()
    data_pie = [data_pie_1, data_pie_2]
    plt.axis('equal')
    Titre = 'Répartition Aliment sain/Aliment pas sain pour nb_sain = %s ' % nb_sain
    plt.title(Titre)
    plt.pie(data_pie, labels=['Sain', 'Pas Sain'], shadow=True, explode=[0, 0.1], autopct='%1.1f%%')
    plt.show()

def main():
    """
        Note : La première colonne et la dernière ont un " caché
    """

    # On charge le dataset sur les colonnes qui nous ont intéressés dans la
    # fonction du dessus
    data = pd.read_csv(_FICHIER,
                       error_bad_lines=False,
                       engine='python',
                       sep=',')

    # On supprime une colonne inutile
    del data['Unnamed: 0']

    # Transposition du dataframe de données pour l'abalyse univariée
    fichier_save = _DOSSIERTRAVAIL + '\\' + 'transposition.csv'
    data_transpose = data.describe().reset_index().transpose()
    print (data_transpose)
    data_transpose.to_csv(fichier_save)

    # Création des histogrammes
    for colon in data:
        if colon != 'nutrition_grade_fr':
            histogramme(data, colon)

    # Création du collérogramme
    correlation_matrix(data)

    # Création des régressions linéaires
    for colon in data:
        if colon != 'nutrition_grade_fr' and colon != 'nutrition-score-fr_100g':
            regression_lineaire(data, 'nutrition-score-fr_100g', colon)

    # Autres régressions intéressantes
    regression_lineaire(data, 'energy_100g', 'fat_100g')
    regression_lineaire(data, 'energy_100g', 'fiber_100g')
    regression_lineaire(data, 'fat_100g', 'saturated-fat_100g')
    regression_lineaire(data, 'sugars_100g', 'salt_100g')
    regression_lineaire(data, 'proteins_100g', 'calcium_100g')

    # Variables tableaux qui vont être utilisées
    results = []
    res_a = []
    res_b = []
    res_c = []
    res_d = []
    res_e = []

    # Copy de la database initial pour ne pas travailler dessus directement
    df = data.copy()

    # Suppresion de la colonne non-chiffre
    del df['nutrition_grade_fr']

    # On va scaler les données pour les prochaines colonnes créées
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df.values)
    df2 = pd.DataFrame(x_scaled)

    # On recopie les noms des colonnes
    df2.columns = df.columns

    # Calculs
    # Somme des élments positifs
    df2['Positif'] = df2['vitamin-a_100g']+df2['vitamin-c_100g']+df2['fiber_100g']+df2['proteins_100g']+df2['iron_100g']

    # Somme des éléments négatifs
    df2['Negatif'] = df2['sugars_100g']+df2['salt_100g']+df2['energy_100g']+df2['fat_100g']

    # Différence du résultat
    df2['Diff'] = (0.01+df2['Positif'])/(0.01+df2['Negatif'])

    # On fait une boucle pour faire les 3 à la suite
    nom_colonne = ['Positif', 'Negatif', 'Diff']
    choices = ['e', 'd', 'c', 'b', 'a']

    # Taille de l'absicce
    array_x_plot = np.arange(10.0, 300.0, 2)

    # de i=1 à i+200 avec i=i+1
    for i in array_x_plot:

        for colonne in nom_colonne:
            # Valeur max de la colonne
            value_max = df2[colonne].max()

            # On divise en 5 subsets
            conditions = [
                (df2[colonne] <= (1/i)*(value_max)),
                (df2[colonne] > (1/i)*(value_max)) & (df2[colonne] <= (2/i)*(value_max)),
                (df2[colonne] > (2/i)*(value_max)) & (df2[colonne] <= (3/i)*(value_max)),
                (df2[colonne] > (3/i)*(value_max)) & (df2[colonne] <= (4/i)*(value_max)),
                (df2[colonne] > (4/i)*(value_max))]

            # On rajoute la colonne avec la donnée
            colonne_cible = 'Indice' + colonne[0]
            df2[colonne_cible] = np.select(conditions, choices)

        # On recopie une colonne de l'autre dataset
        df['Qualite'] = df2['IndiceD']

        # Rajout de la colonne du nutri score en lettres
        df['nutrition_grade_fr'] = data['nutrition_grade_fr']

        # Rajout d'une colonne qui est True/False si les valeurs sont égales
        df['Verdict'] = df['Qualite'] == df['nutrition_grade_fr']

        # Nom de la colonne de référence
        ref = 'nutrition_grade_fr'

        # Calcul des scores
        score = 100 * df['Verdict'].where(df['Verdict']).count()/df['Verdict'].count()
        results.append(score)

        s_a = df['Qualite'].where(df['Qualite'] == 'a') == df[ref].where(df[ref] == 'a')
        s_a = 100 * s_a.where(s_a).count()/df[ref].where(df[ref] == 'a').count()
        res_a.append(s_a)

        s_b = df['Qualite'].where(df['Qualite'] == 'b') == df[ref].where(df[ref] == 'b')
        s_b = 100 * s_b.where(s_b).count()/df[ref].where(df[ref] == 'b').count()
        res_b.append(s_b)

        s_c = df['Qualite'].where(df['Qualite'] == 'c') == df[ref].where(df[ref] == 'c')
        s_c = 100 * s_c.where(s_c).count()/df[ref].where(df[ref] == 'c').count()
        res_c.append(s_c)

        s_d = df['Qualite'].where(df['Qualite'] == 'd') == df[ref].where(df[ref] == 'd')
        s_d = 100 * s_d.where(s_d).count()/df[ref].where(df[ref] == 'd').count()
        res_d.append(s_d)

        s_e = df['Qualite'].where(df['Qualite'] == 'e') == df[ref].where(df[ref] == 'e')
        s_e = 100 * s_e.where(s_e).count()/df[ref].where(df[ref] == 'e').count()
        res_e.append(s_e)

        # print('Pour i =', i, '\tBonnes valeurs : ', round(score, 3), '%')

    plt.figure(figsize=(12, 8))
    plt.title('Taux de matchs en fonction de i')
    plt.plot(array_x_plot, results)

    # Affichage du max
    ymax = max(results)
    xpos = results.index(ymax)
    xmax = array_x_plot[xpos]
    plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax-5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.grid('on')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.grid('on')
    plt.title('Rapport des A, B, C, D, E')
    plt.plot(array_x_plot, res_a, color='red')
    plt.plot(array_x_plot, res_b, color='pink')
    plt.plot(array_x_plot, res_c, color='blue')
    plt.plot(array_x_plot, res_d, color='black')
    plt.plot(array_x_plot, res_e, color='green')

    # Création des légendes
    p_a = mpatches.Patch(color='red', label='a')
    p_b = mpatches.Patch(color='pink', label='b')
    p_c = mpatches.Patch(color='blue', label='c')
    p_d = mpatches.Patch(color='black', label='d')
    p_e = mpatches.Patch(color='green', label='e')
    plt.legend(handles=[p_a, p_b, p_c, p_d, p_e])

    plt.show()

    # Création du camembert
    # 3 cas possibles
    # Dictionnaire pour décider des bons aliments ou pas
    dico = {'a': True, 'b': False, 'c' : False, 'd' : False, 'e' : False}
    sanite(df, dico, 1)

    dico = {'a': True, 'b': True, 'c' : False, 'd' : False, 'e' : False}
    sanite(df, dico, 2)

    dico = {'a': True, 'b': True, 'c' : True, 'd' : False, 'e' : False}
    sanite(df, dico, 3)

if __name__ == "__main__":
    main()
    