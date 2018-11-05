# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 19:57:19 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

# Pour ne pas avoir les warnings lors de la compilation
warnings.filterwarnings("ignore")

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p5\\'
_DOSSIERIMAGE = 'C:\\Users\\Toni\\python\\python\\Projet_5\\images'
_FICHIERDATAKMEANS = _DOSSIER + 'dataset_p5_v2a.csv'
_FICHIERDATARFC = _DOSSIER + 'dataset_p5_v2b.csv'

_VERBOSE = 0

def plot_confusion_matrix(cm, classes, title, biais):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Taille de la figure
    np.set_printoptions(precision=2)
    plt.figure(figsize=(7, 7))

    # Jeu de couleur
    cmap = plt.cm.Greens

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    title = "Matrix_" + biais + "_"  + title
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(_DOSSIERIMAGE + "\\" + title)
    plt.show()

def appel_cvs(xtrain, xtest, ytrain, ytest, biais):
    """
    Fonction qui fournit les hyperparamètres aux modèles et appelle les fonctions
    """

    # Choix de l'algorithme de classification
    model = [KNeighborsClassifier(),
             AdaBoostClassifier(),
             RandomForestClassifier(),
             GradientBoostingClassifier()
            ]

    # Hyperparamètres
    param_grid = [{'n_neighbors': [5, 9, 13, 17, 23, 29, 37, 43, 50]},
                  {'n_estimators': [5, 20, 35, 50, 65]},
                  {'max_depth': [None, 10, 20, 30], 'n_estimators': [5, 20, 35, 50]},
                  {'max_depth': [None, 10, 20, 30], 'n_estimators': [5, 20, 35, 50]}
                 ]

    # Appel de fonction avec le RandomForestRegressor
    for i in range(0, len(model)):
        log_cv = algos_cv(xtrain, xtest, ytrain, ytest, model[i], param_grid[i], biais)

def algos_cv(xtrain, xtest, ytrain, ytest, model, param_grid, biais):
    """
    TBD
    """

    # Score à améliorer
    score = 'accuracy'

    print(model.__class__.__name__, "\n")

    # Options de l'algorithme
    clf = GridSearchCV(model,
                       param_grid=param_grid,
                       verbose=_VERBOSE,
                       cv=5,
                       scoring=score,
                       refit=True,
                       return_train_score=False)

    # Fit
    clf.fit(xtrain, ytrain)

    # Liste qui va garder les résultats
    log_cols = ["Accuracy", "Hyperparametres"]
    log_cv = pd.DataFrame(columns=log_cols)

    # Affichages
    for score, params in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['params']):
        print("Score : ", round(score*100, 2), "pour", params)

        # Sauvegarde des scores de predictions
        log_entry = pd.DataFrame([[round(score*100, 2), params]], columns=log_cv.columns)
        log_cv = log_cv.append(log_entry)

    # Meilleurs paramètres
    score_max = round(clf.best_score_*100, 2)
    print("Meilleur score : ", score_max, "pour", clf.best_params_, "\n")

    # Affichage du diagramme en baton
    affichage_score(model, log_cv, biais)

    ypred = clf.best_estimator_.predict(xtest)

    # Affichage de la matrice de confusion
    cnf_matrix = confusion_matrix(ypred, ytest, labels=ytrain.unique())
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix, ytest.unique(), model.__class__.__name__, biais)

    return log_cv

def affichage_score(model, log_cv, biais):
    """
    Diagrammes en batons pour voir les résultats
    """

    # Mise en forme légère
    log_cv = log_cv.reset_index()
    del log_cv['index']

    # Noms des variables
    data_colonne = log_cv['Accuracy']
    data_ligne = log_cv['Hyperparametres']

    # La figure change de taille suivant le nombre de données
    plt.figure(figsize=(len(data_colonne), 8))

    # Données de l'axe X
    x_axis = [k for k, i in enumerate(data_colonne)]
    x_label = [i for i in data_ligne]

    # Données de l'axe Y
    y_axis = [i for i in data_colonne]

    # Limite de l'axe Y
    plt.ylim(min(log_cv['Accuracy'])-0.5, max(log_cv['Accuracy'])+0.5)

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
    plt.ylabel('Accuracy')
    titre = 'Accuracy pour ' + model.__class__.__name__ + ", " + "biais = " + biais
    plt.title(titre)
    plt.tight_layout()
    plt.savefig(_DOSSIERIMAGE + "\\Accuracy_" + biais + "_"  + model.__class__.__name__)
    plt.show()

def algo_wo_optimisation(xtrain, xtest, ytrain, ytest):
    """
    Tests de différentes algorithems sans optimisation recherchée
    Uniquement pour avoir une petite idée de ce qu'ils sont capables de faire
    """

    classifiers = [KNeighborsClassifier(3),
                   RandomForestClassifier(),
                   AdaBoostClassifier(),
                   GradientBoostingClassifier(),
                  ]

    # Logging for Visual Comparison
    log_cols = ["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)

    for clf in classifiers:
        clf.fit(xtrain, ytrain)
        name = clf.__class__.__name__

        # Affichage
        print("="*30)
        print(name)
        print('****Resultats****')

        # Scores des prédictions
        train_predictions = clf.predict(xtest)
        acc = accuracy_score(ytest, train_predictions)
        print("Accuracy: {:.4%}".format(acc))

        # Scores des prédictions
        train_predictions = clf.predict_proba(xtest)
        logloss = log_loss(ytest, train_predictions)
        print("Log Loss: {}".format(logloss))

        log_entry = pd.DataFrame([[name, acc*100, logloss]], columns=log_cols)
        log = log.append(log_entry)

    print("="*30)

    # Graphiques de comparaison
    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

    plt.xlabel('Accuracy %')
    plt.title('Classifier Accuracy')
    plt.show()

    sns.set_color_codes("muted")
    sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

    plt.xlabel('Log Loss')
    plt.title('Classifier Log Loss')
    plt.show()

def main():
    """
    Fonction principale
    """

    # Lecture du dataset
    data = pd.read_csv(_FICHIERDATARFC, error_bad_lines=False)

    # Récupération de l'index
    data = data.set_index('Unnamed: 0')

    # Axe X
    data_x = data.copy()

    # On supprime les étiquettes de l'axe X
    del data_x['labels']

    # Axe Y = étiquettes
    data_y = data['labels']

    # Essai dans biais dans les données
    # Répartition Train/Test
    xtrain, xtest, ytrain, ytest = train_test_split(data_x, data_y, train_size=0.75, stratify=data_y)

    # Sans optimisations
    algo_wo_optimisation(xtrain, xtest, ytrain, ytest)

    # Recherche d'optimisation
    appel_cvs(xtrain, xtest, ytrain, ytest, "aucun")
