#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 21:38:27 2018

@author: toni
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.cluster.vq import whiten
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
import cv2

# Pour ne pas avoir les warnings lors de la compilation
warnings.filterwarnings("ignore")

# Lieu où se trouvent des images
IMG_DIR = '/home/toni/Bureau/p7/Images/'
DOSSIER_SAVE = '/home/toni/python/python/Projet_7/images/'

# Définitions des limites d'execution
NB_RACES = 5
NB_EXEMPLES = 200
NB_CLUSTER = int(NB_RACES * (NB_EXEMPLES/5))
AFFICHAGE_HISTOGRAMME = True
RESULTATS = pd.DataFrame()

# Setup a standard image size
STANDARD_SIZE = (224, 224) #(300, 167)

def gestion_erreur(res, test_y, labels, classifieur):
    """
    Gestion de l'erreur quand une catégorie de chien n'est pas prédite
    On rajoute la colonne vide manuellement
    """

    # Si ce n'est pas un kmeans, le traitement est différent (noms ou numéros)
    if classifieur != 'kmeans':
        for i in np.unique(test_y):
            if i not in res.columns:
                res[i] = 0

        for i in res.columns:
            if i not in res.index:
                res.loc[i] = 0
    else:
        for i in range(0, NB_RACES):
            if i not in res.columns:
                res[i] = 0

        for i in np.unique(labels):
            if i not in res.index:
                res.loc[i] = 0

    res = res.sort_index(axis=0, ascending=True)
    res = res.sort_index(axis=1, ascending=True)

    return res

def fonction_median(img, param1):
    """
    Fonction de filtre
    """

    # Application du filtre
    img_modified = scipy.ndimage.median_filter(img, size=param1)

    return img_modified

def fonction_gauss(img, param1):
    """
    Fonction de filtre
    """

    # Application du filtre
    img_modified = scipy.ndimage.filters.gaussian_filter(img, sigma=param1)

    return img_modified

def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """

    img = Image.open(filename)

    if verbose:
        print("changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE)))

    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = np.array(img)
    return img

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it
    into an array of shape (1, m * n)
    """

    shape = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, shape)
    return img_wide[0]

def recup_images_filtres(liste_images, num_filtre):
    """
    Fonction qui récupére toute les images avec une sélection aléatoire
    Rajout de filtres possibles
    """

    # Création des listes vides
    data = []

    for lien_image in liste_images:
        # Récupération de la matrice tranformée
        img = img_to_matrix(lien_image, False)

        if num_filtre == 1:
            # Filtre gaussien
            img = fonction_gauss(img, 5)
        elif num_filtre == 2:
            # Filtre médian
            img = fonction_median(img, 5)
        elif num_filtre == 3:
            img = whiten(img)

        # Mise à une dimension
        img = flatten_image(img)
        data.append(img)

        del img

    return data

def features(img, extractor):
    """
    Detect and compute interest points and their descriptors.
    """

    img = cv2.imread(img)
    img = cv2.resize(img, STANDARD_SIZE)

    #img = img.resize(STANDARD_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, des = extractor.detectAndCompute(img, None)

    return keypoints, des

def calcul_resultats(res, test_y, classifieur, nom_filtre):
    """
    Fonction qui va calculer les pourcentages de bons pronostics
    """

    global RESULTATS

    print("\nResultats pour", classifieur)

    # Transformation en tableau exploitable
    res1 = res.values

    data_resultats = pd.DataFrame(index=res.index, columns=['bons',
                                                            'prono',
                                                            'total',
                                                            'pc_prono',
                                                            'pc_total'])

    # Affichage des résultats
    print("Resultat :", round(100*res1.diagonal().sum()/len(test_y), 2), "%")
    print("No d'erreurs =", len(test_y) - res1.diagonal().sum(), "/", len(test_y))

    for i in range(0, len(res)):
        diagonale = res1.diagonal()[i]
        data_resultats.loc[res.index[i], 'bons'] = diagonale
        data_resultats.loc[res.index[i], 'prono'] = res.sum()[i]
        data_resultats.loc[res.index[i], 'total'] = res.sum('columns')[i]
        data_resultats.loc[res.index[i], 'pc_prono'] = round(100*diagonale/res.sum()[i], 2)
        data_resultats.loc[res.index[i], 'pc_total'] = round(100*diagonale/res.sum('columns')[i], 2)

    data_resultats = data_resultats.fillna(0)

    temp = []
    temp.append([classifieur,
                 nom_filtre,
                 data_resultats['pc_prono'].mean(),
                 data_resultats['pc_total'].mean()])

    RESULTATS = RESULTATS.append(temp)

    print(data_resultats)

def fonction_orb(liste_images):
    """
    Fonction qui extrait les features et permets de les clusterizer
    """

    # Création des listes vides
    pool_descriptors = []

    #
    extractor = cv2.ORB_create()

    for lien_image in liste_images:
        # Récupération de la matrice tranformée
        keypoints, descriptors = features(lien_image, extractor)

        # Rajout à la liste
        pool_descriptors.append(descriptors)

    # Mise au bon format
    pool_descriptors = np.asarray(pool_descriptors)
    pool_descriptors = np.concatenate(pool_descriptors, axis=0)

    # Clusturisation des descriptors
    print("Training MiniBatchKMeans")
    kmeans = MiniBatchKMeans(n_clusters=NB_CLUSTER).fit(pool_descriptors)
    print("End training MiniBatchKMeans")

    return kmeans

def etablir_liste_chiens():
    """
    Création de la liste aléatoire des chiens pour les races selectionnés
    """

    # Listes
    liste_dossier = []
    liste_images = []
    labels = []

    # Valeur initiale d'un compteur
    cpt_race = 0

    # Création de la liste aléatoire des races
    liste_chiens = os.listdir(IMG_DIR)

    for i in range(0, NB_RACES):
        nb_alea = random.randrange(0, len(liste_chiens))
        liste_dossier.append(liste_chiens[nb_alea])
        del liste_chiens[nb_alea]

    # Création de la liste aléatoire des chiens pour les races selectionnés
    for dirs in liste_dossier:
        # Valeur initiale d'un compteur
        cpt_exemple = 0
        if cpt_race < NB_RACES+1:
            cpt_race = cpt_race+1
            for filename in os.listdir(IMG_DIR + dirs):
                # On ne garde que NB_EXEMPLES exemplaires de chaque race
                if cpt_exemple < NB_EXEMPLES:
                    cpt_exemple = cpt_exemple+1

                    # Chemin complet de l'image
                    liste_images.append(IMG_DIR + dirs + '/' + filename)

                    # Rajout du label
                    labels.append(dirs[dirs.find('-')+1:].lower())

    return liste_images, labels

def calculate_centroids_histogram(liste_images, labels, model):
    """
    with the k-means model found, this code generates the feature vectors
    by building an histogram of classified keypoints in the kmeans classifier
    """

    # Création des listes vides
    feature_vectors = []
    class_vectors = []
    compteur = 0

    # Extracteur de features
    extractor = cv2.ORB_create() #cv2.xfeatures2d.SIFT_create()

    for lien_image in liste_images:
        # Récupération de la matrice tranformée
        keypoints, descriptors = features(lien_image, extractor)

        # classification of all descriptors in the model
        predict_kmeans = model.predict(descriptors)

        # calculates the histogram
        hist, bin_edges = np.histogram(predict_kmeans, bins=NB_CLUSTER)

        # Affichage des histogrammes
        if AFFICHAGE_HISTOGRAMME and compteur < 6:
            compteur = compteur + 1
            plt.hist(hist, bins=len(bin_edges), align='mid')
            plt.xlabel('bins')
            plt.ylabel('valeurs')
            plt.title('Histogramme')
            plt.tight_layout()
            plt.savefig(DOSSIER_SAVE + 'ex' + str(compteur) + '_histogramme', dpi=100)
            plt.show()

        # histogram is the feature vector
        feature_vectors.append(hist)

    # Mise sous la bonne forme
    feature_vectors = np.asarray(feature_vectors)
    class_vectors = np.asarray(labels)

    # return vectors and classes we want to classify
    return class_vectors, feature_vectors

def fonction_bovw(liste_images, labels):
    """
    Fonction avec la technique bag of visual words
    """

    print("\nFiltre ORB")

    # Séparation des datasets testing/training
    train_x, test_x, train_y, test_y = train_test_split(liste_images,
                                                        labels,
                                                        test_size=0.25)

    # Entrainement du modèle sur le dataset de training
    trained_model = fonction_orb(train_x)

    # Extraction des histogrammes
    [train_class, train_featvec] = calculate_centroids_histogram(train_x,
                                                                 train_y,
                                                                 trained_model)
    [test_class, test_featvec] = calculate_centroids_histogram(test_x,
                                                               test_y,
                                                               trained_model)

    # Utilisation des vecteurs de training pour entrainer le classifieur
    clf = svm.SVC()
    clf.fit(train_featvec, train_class)
    predict = clf.predict(test_featvec)

    # Calcul des résultats
    res = pd.crosstab(np.asarray(test_class),
                      predict,
                      rownames=["Actual"],
                      colnames=["Predicted"])

    # Gestion d'une erreur
    if len(res.columns) != NB_RACES:
        res = gestion_erreur(res, test_y, labels, 'svm')
    calcul_resultats(res, test_y, 'svm', 'orb')

    # Test avec KNN()
    knn = KNeighborsClassifier(n_neighbors=50)
    knn.fit(train_featvec, train_class)
    predict = knn.predict(test_featvec)

    # Calcul des résultats
    res = pd.crosstab(np.asarray(test_class),
                      predict,
                      rownames=["Actual"],
                      colnames=["Predicted"])

    # Gestion d'une erreur
    if len(res.columns) != NB_RACES:
        res = gestion_erreur(res, test_y, labels, 'knn')
    calcul_resultats(res, test_y, 'knn', 'orb')

def fonction_filtres(liste_images, labels):
    """
    Fonction avec filtres traditionnels
    """

    for num_filtre in range(0, 4):
        if num_filtre == 0:
            nom_filtre = "Aucun"
        elif num_filtre == 1:
            nom_filtre = "Gaussien"
        elif num_filtre == 2:
            nom_filtre = "Median"
        elif num_filtre == 3:
            nom_filtre = "Whitening"

        print("\nFiltre", nom_filtre)
        data = recup_images_filtres(liste_images, num_filtre)

        ## Réduction de dimension
        # PCA
        #pca = RandomizedPCA(n_components=2)
        #data = pca.fit_transform(data)
        #nom_reducteur = 'pca'
        # Explication de la variance
        #print(pca.explained_variance_ratio_)

        # t-SNE
        data = TSNE(n_components=2).fit_transform(data, labels)
        nom_reducteur = 'tsne'

        # Affichage en 2D après une décomposition
        affichage_decomposition(data, labels, nom_reducteur, nom_filtre)

        # Séparation des datasets testing/training
        train_x, test_x, train_y, test_y = train_test_split(data,
                                                            labels,
                                                            test_size=0.25)

        # Transformation en array
        test_y = np.array(test_y)
        train_y = np.array(train_y)

        ## Création de la méthode de classification
        # Test avec KNN
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(train_x, train_y)
        res = pd.crosstab(test_y,
                          knn.predict(test_x),
                          rownames=["Actual"],
                          colnames=["Predicted"])

        # Gestion d'une erreur
        if len(res.columns) != NB_RACES:
            res = gestion_erreur(res, test_y, '0', 'knn')
        calcul_resultats(res, test_y, 'knn', nom_filtre)

        # Test avec Kmeans
        kmeans = KMeans(n_clusters=NB_RACES).fit(train_x, train_y)
        res = pd.crosstab(test_y,
                          kmeans.predict(test_x),
                          rownames=["Actual"],
                          colnames=["Predicted"])

        # Gestion d'une erreur
        if len(res.columns) != NB_RACES:
            res = gestion_erreur(res, test_y, labels, 'kmeans')
        calcul_resultats(res, test_y, 'kmeans', nom_filtre)

def affichage_decomposition(data, labels, nom_reducteur, nom_filtre):
    """
    Affichage en 2D de la décomposition"
    """

    principaldf = pd.DataFrame(data=data,
                               columns=['principal component 1',
                                        'principal component 2'])

    finaldf = pd.concat([principaldf, pd.DataFrame(labels)], axis=1)

    data_labels = pd.DataFrame(labels)
    targets = []

    # Création de la liste des labels
    for i in data_labels[0].unique():
        targets.append(i)

    fig = plt.figure(figsize=(8, 8))
    axe = fig.add_subplot(1, 1, 1)
    axe.set_xlabel('Principal Component 1', fontsize=15)
    axe.set_ylabel('Principal Component 2', fontsize=15)
    title = nom_reducteur + '_components'
    axe.set_title(title, fontsize=20)

    # Création de la liste des couleurs
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(targets)))

    # Affichage par catégorie
    for target, color in zip(targets, colors):
        indices_to_keep = finaldf[0] == target
        axe.scatter(finaldf.loc[indices_to_keep, 'principal component 1'],
                    finaldf.loc[indices_to_keep, 'principal component 2'],
                    c=color,
                    s=50)

    axe.legend(targets)
    axe.grid()
    plt.tight_layout()
    plt.savefig(DOSSIER_SAVE + nom_reducteur + '_' + nom_filtre, dpi=100)
    plt.show()

def main(choix):
    """
    Fonction principale
    """

    # Etablir la liste des chiens
    liste_images, labels = etablir_liste_chiens()

    if choix == 0:
        fonction_filtres(liste_images, labels)
    elif choix == 1:
        fonction_bovw(liste_images, labels)
    else:
        fonction_filtres(liste_images, labels)
        fonction_bovw(liste_images, labels)

    RESULTATS.columns= ["nom", "nom2", "% prono", "% total"]

    print(RESULTATS)
#-----
#    # Création du dataframe vide
#    spec_images = pd.DataFrame()

    # Partie pour récupérer les tailles des images.
    # Pas forcément utile
#    for path, dirs, files in os.walk(img_dir):
#        for filename in files:
#            image = misc.imread(path + '/' + filename)
#            titre = path + '/' + filename
#            spec_images.loc[titre, 0] = image.shape[0]
#            spec_images.loc[titre, 1] = image.shape[1]
#            spec_images.loc[titre, 2] = image.shape[2]
#            spec_images.loc[titre, 3] = str(image.shape[0]) + '-' + \
#                                        str(image.shape[1]) + '-' + \
#                                        str(image.shape[2])
#
#    df = spec_images.groupby(3)[0].count()
