# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 21:12:42 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
from math import sqrt
import warnings
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.externals import joblib

# Pour ne pas avoir les warnings lors de la compilation
warnings.filterwarnings("ignore")

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p4\\'
_DOSSIERPKL = 'C:\\Users\\Toni\\python\\python\\Projet_4\\pkl'
_DOSSIERIMAGE = 'C:\\Users\\Toni\\python\\python\\Projet_4\\images'
_FICHIERDATA = _DOSSIER + 'dataset_p4.csv'
_VERBOSE = 10

# Booléean pour faire la différence entre un fit et un joblib load
_RECALCUL_JOBLIB = False
_RECALCUL_JOBLIB_HYP = False

def main():
    """
    Fonction main
    """

    # Récupération des dataset
    data = pd.read_csv(_FICHIERDATA, error_bad_lines=False, low_memory=False)
    del data['Unnamed: 0']

    # Premier algorithme
    log, log_cv = lancer_algorithme(data)

    # Affichages
    print(log)
    print(log_cv)

    # On enlève un regresseur hors-norme
    #log = log[log['Classifier'] != 'LinearRegression']

    # Affichages
    affichage_resultats(log.pivot(index='Id', columns='Classifier', values='RMSE'))
    affichage_resultats(log.pivot(index='Classifier', columns='Id', values='RMSE'))

    # Modèle retenu :
    # SGDRegressor
    #
    # Avec les hyperparamètres suivants :
    #
    # alpha = 0,001
    # l1_ratio = 0,5
    # penalty = ‘none’

def lancer_algorithme(data):
    """
    Fonction de calcul pour tous les algorithmes différents pour chaque compagnie
    """

    # Logging for Visual Comparison
    log_cols = ["Classifier", "Id", "RMSE", "R2"]
    log = pd.DataFrame(columns=log_cols)

    # Création de la liste (unique) des compagnies aériennes
    liste = data['UNIQUE_CARRIER'].unique()
    
    for compagnie in liste:

        # Copie de sauvegarde
        datanum = data.copy()

        print('\n\nPour la compagnie', compagnie)

        # On ne garde que les données pour 1 compagnie à la fois
        datanum = datanum[datanum['UNIQUE_CARRIER'] == compagnie]

        # Puis on supprime cette donnée car on sait qu'elle sera toujours la même
        del datanum['UNIQUE_CARRIER']

        # Axe X
        data_x = datanum.copy()

        # On supprime les étiquettes de l'axe X
        del data_x['ARR_DELAY']

        # Axe Y
        data_y = datanum['ARR_DELAY']

        # One-Hot encoding
        liste_criteres = ['ORIGIN',
                          'DEP_TIME_BLK',
                          'DAY_OF_WEEK',
                          'DEST',
                          'MONTH']
        data_x = pd.get_dummies(data=data_x, columns=liste_criteres)

        # On supprime les nan
        data_x.fillna(0, inplace=True)
        data_y.fillna(0, inplace=True)

        # Répartition Train/Test
        xtrain, xtest, ytrain, ytest = train_test_split(data_x, data_y, train_size=0.75)

        # Fonction qui va comparer les algorithmes sans optimisations
        log = algo_wo_optimisation(xtrain, xtest, ytrain, ytest, compagnie, log)

        # Fonction qui permets de faire les CV
        log_cv = appel_cvs(xtrain, ytrain, compagnie)

    return log, log_cv

def algo_wo_optimisation(xtrain, xtest, ytrain, ytest, compagnie, log):
    """
    Tests de différentes algorithems sans optimisation recherchée
    Uniquement pour avoir une petite idée de ce qu'ils sont capables de faire
    """

    classifiers = [SGDRegressor(),
                   #AdaBoostRegressor(),
                   #LinearRegression(),
                   ElasticNetCV(),
                   LassoCV(),
                   #OrthogonalMatchingPursuitCV(),
                   RidgeCV(),
                   RandomForestRegressor()
                   ]
                           
    for clf in classifiers:

        # Nom du classifieur
        name = clf.__class__.__name__

        # Localisation de du fichier du fit sauvegardé
        fichier = _DOSSIERPKL + "\\" + name + "_" + compagnie + ".pkl"

        # Choix entre fit de nouveau ou aller chercher le fit sauvegardé
        if _RECALCUL_JOBLIB is True:
            # Fit
            clf.fit(xtrain, ytrain)
            # Dump (sauvegarde)
            joblib.dump(clf, fichier)
        else:
            # On va chercher le dump
            clf = joblib.load(fichier)

        print("="*40)
        print(name)

        # Predictions
        train_predictions = clf.predict(xtest)

        # Scores des prédictions
        mse = sqrt(abs(mean_squared_error(ytest, train_predictions)))
        score2 = 100 * r2_score(ytest, train_predictions)

        # Affichage des scores de prédictions
        print("RMSE : ", round(mse, 4))
        print("R2 : ", round(score2, 3))

        # Sauvegarde des scores de predictions
        log_entry = pd.DataFrame([[name, compagnie, mse, score2]], columns=log.columns)
        log = log.append(log_entry)

    return log

def appel_cvs(xtrain, ytrain, compagnie):
    """
    Fonction qui fournit les hyperparamètres aux modèles et appelle les fonctions
    """

    # Choix de l'algorithme de régression : SGDRegressor et hyperparamètres
    model = SGDRegressor()
    param_grid = [{'alpha' : 10.0**-np.arange(1, 5),
                   'l1_ratio':[0, .1, .5, .9, 1],
                  }]

    # Appel de fonction avec le SGDRegressor
    log_cv = algos_cv(xtrain, ytrain, model, param_grid, compagnie)

     # Choix de l'algorithme de régression : Ridge et hyperparamètres
    model = Ridge()
    param_grid = {'alpha': np.logspace(-7, 7, 15)}

    # Appel de fonction avec le Ridge
    log_cv = algos_cv(xtrain, ytrain, model, param_grid, compagnie)

    # Choix de l'algorithme de régression RFR et hyperparamètres
    model = RandomForestRegressor()
    param_grid = {'max_depth': [None, 10, 20],
                  'n_estimators': [5, 20, 35]}

    # Appel de fonction avec le RandomForestRegressor
    log_cv = algos_cv(xtrain, ytrain, model, param_grid, compagnie)

    # Choix de l'algorithme de régression Lasso et hyperparamètres
    model = LassoCV()
    param_grid = {'eps': [1e-1, 1e-2, 1e-3],
                  'n_alphas': range(2, 6)}

    # Appel de fonction avec le RandomForestRegressor
    log_cv = algos_cv(xtrain, ytrain, model, param_grid, compagnie)

    return log_cv

def algos_cv(xtrain, ytrain, model, param_grid, compagnie):
    """
    TBD
    """

    # Score à améliorer
    score = 'neg_mean_squared_error'

    # Options de l'algorithme
    clf = GridSearchCV(model,
                       param_grid=param_grid,
                       verbose=_VERBOSE,
                       cv=5,
                       scoring=score,
                       refit=True,
                       return_train_score=False)

    # Localisation de du fichier du fit sauvegardé
    fichier = _DOSSIERPKL + "\\Hyp_" + model.__class__.__name__ + "_" + compagnie + ".pkl"

    # Choix entre fit de nouveau ou aller chercher le fit sauvegardé
    if _RECALCUL_JOBLIB_HYP is True:
        # Fit
        clf.fit(xtrain, ytrain)
        # Dump du fichier
        joblib.dump(clf, fichier)
    else:
        # On va chercher le dump
        clf = joblib.load(fichier)

    # Liste qui va garder les résultats
    log_cols = ["RMSE", "Hyperparametres"]
    log_cv = pd.DataFrame(columns=log_cols)

    # Affichages
    for mse, params in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['params']):
        print("RMSE : ", round(sqrt(abs(mse)), 4), "pour", params)

        # Sauvegarde des scores de predictions
        log_entry = pd.DataFrame([[sqrt(abs(mse)), params]], columns=log_cv.columns)
        log_cv = log_cv.append(log_entry)

    # Meilleurs paramètres
    score_max = round(sqrt(abs(clf.best_score_)), 4)
    print("\nMeilleur score : ", score_max, "pour", clf.best_params_)

    # Affichage du diagramme en baton
    affichage_rmse(model, log_cv, compagnie)

    return log_cv

def affichage_rmse(model, log_cv, compagnie):
    """
    Diagrammes en batons pour voir les résultats
    """

    # Mise en forme légère
    log_cv = log_cv.reset_index()
    del log_cv['index']

    # Noms des variables
    data_colonne = log_cv['RMSE']
    data_ligne = log_cv['Hyperparametres']

    # La figure change de taille suivant le nombre de données
    plt.figure(figsize=(len(data_colonne), 8))

    # Données de l'axe X
    x_axis = [k for k, i in enumerate(data_colonne)]
    x_label = [i for i in data_ligne]

    # Données de l'axe Y
    y_axis = [i for i in data_colonne]

    # Limite de l'axe Y
    plt.ylim(min(log_cv['RMSE'])-0.5, max(log_cv['RMSE'])+0.5)

    # Largeur des barres
    width = 0.2

    # Légende de l'axe X
    plt.xticks(x_axis, x_label, rotation=90)

    # Création
    rects = plt.bar(x_axis, y_axis, width, color='b')

    # On fait les labels pour les afficher
    labels = ["%.2f" % i for i in data_colonne]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        width = rect.get_width()

        plt.text(rect.get_x()+ width/2, height + 0.1, label, ha='center', va='bottom')

    # Barres horizontales
    plt.axhline(y=sum(data_colonne)/len(data_colonne), color='r', linestyle='-')
    plt.axhline(y=min(data_colonne), color='g', linestyle='-')

    # Esthétisme
    plt.grid()
    plt.ylabel('RMSE')
    titre = 'RMSE suivant les hyperparamètres pour ' + compagnie
    plt.title(titre)
    plt.tight_layout()
    plt.savefig(_DOSSIERIMAGE + "\\_RMSE_" + model.__class__.__name__ + "_" + compagnie)
    plt.show()

def affichage_resultats(log_pivot):
    """
    Diagrammes en batons pour voir les résultats
    """

    for nom_colonne in log_pivot:
        plt.figure(figsize=(12, 8))

        data_colonne = log_pivot[nom_colonne]

        # Données de l'axe X
        x_axis = [k for k, i in enumerate(data_colonne)]
        x_label = [i for i in data_colonne.index]

        # Données de l'axe Y
        y_axis = [i for i in data_colonne]

        # Largeur des barres
        width = 0.2

        # Légende de l'axe X
        plt.xticks(x_axis, x_label, rotation=90)

        # Création
        rects = plt.bar(x_axis, y_axis, width, color='b')

        # On fait les labels pour les afficher
        labels = ["%.2f" % i for i in data_colonne]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            width = rect.get_width()

            plt.text(rect.get_x()+ width/2, height + 0.1, label, ha='center', va='bottom')

        # Barres horizontales
        plt.axhline(y=sum(data_colonne)/len(data_colonne), color='r', linestyle='-')
        plt.axhline(y=min(data_colonne), color='g', linestyle='-')

        # Esthétisme
        plt.grid()
        plt.ylabel('RMSE')
        plt.title(nom_colonne)
        plt.tight_layout()
        plt.savefig(_DOSSIERIMAGE + "//_Bar_" + nom_colonne)
        plt.show()
