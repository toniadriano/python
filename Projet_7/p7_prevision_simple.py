#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 22:39:52 2018

@author: toni
"""

# Librairies
import warnings
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import VGG19
from sklearn.externals import joblib

# Pour ne pas avoir les warnings lors de la compilation
warnings.filterwarnings("ignore")

# Variables globales
DOSSIER_SAVE = '/home/toni/python/python/Projet_7/images/'

def main(fichier):
    """
    Fichier principale
    """

    # On crée un modèle déjà pré entrainé
    model = joblib.load(DOSSIER_SAVE + 'savefit.pkl')
    liste_noms = pd.read_csv(DOSSIER_SAVE + 'savefit.csv')
    del liste_noms['Unnamed: 0']
    liste_noms = liste_noms.columns

    # Visualisation de toutes les couches du modèle
    model.summary()

    # On va chercher l'image
    image = load_img(fichier, target_size=(224, 224))

    # Convertion de l'image
    image = img_to_array(image)

    # Reshape en 4 dimensions
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # Preprocessing
    image = preprocess_input(image)

    # Prediction (des probabilités)
    proba = model.predict(image)
    proba = pd.DataFrame(proba, columns=liste_noms)
    label = np.argmax(proba.loc[0], axis=1)

    # Affichage
    print('Race la plus probable :', label)
