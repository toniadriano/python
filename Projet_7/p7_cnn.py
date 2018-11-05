#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:19:42 2018

@author: toni
"""
import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

# Lieu où se trouvent des images
DOSSIER_SOURCE = '/home/toni/Bureau/p7/flow/'
IMG_DIR = '/home/toni/Bureau/p7/Images/'
DOSSIER_SAVE = '/home/toni/python/python/Projet_7/images/'

# Définitions des limites d'execution
NB_RACES = 10
NB_EXEMPLES = 300
T_IMG = 224
BATCH_SIZE = 32
DATA_AUGMENTATION = False

def cnn_calcul_resultats(res, test_y, classifieur):
    """
    Fonction qui va calculer les pourcentages de bons pronostics
    """

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

    for i in range(0, len(res)):
        diagonale = res1.diagonal()[i]
        data_resultats.loc[res.index[i], 'bons'] = diagonale
        data_resultats.loc[res.index[i], 'prono'] = res.sum()[i]
        data_resultats.loc[res.index[i], 'total'] = res.sum('columns')[i]
        data_resultats.loc[res.index[i], 'pc_prono'] = round(100*diagonale/res.sum()[i], 2)
        data_resultats.loc[res.index[i], 'pc_total'] = round(100*diagonale/res.sum('columns')[i], 2)

    data_resultats = data_resultats.fillna(0)

    print(data_resultats)

def cnn_data_augmentation(model, liste_train, liste_test):
    """
    Fonction pour la data augmentation
    """

    # DA pour le training
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                                                                 width_shift_range=0.2,
                                                                 height_shift_range=0.2,
                                                                 shear_range=0.2,
                                                                 zoom_range=0.2,
                                                                 horizontal_flip=True)

    train_generator = train_datagen.flow_from_dataframe(dataframe=liste_train,
                                                        directory=DOSSIER_SOURCE,
                                                        x_col='liste',
                                                        y_col='labels',
                                                        has_ext=True,
                                                        target_size=(T_IMG, T_IMG),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical')

    # DA pour la validation
    valid_datagen = keras.preprocessing.image.ImageDataGenerator()

    valid_generator = valid_datagen.flow_from_dataframe(dataframe=liste_test,
                                                        directory=DOSSIER_SOURCE,
                                                        x_col='liste',
                                                        y_col='labels',
                                                        has_ext=True,
                                                        target_size=(T_IMG, T_IMG),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical')

    hist = model.fit_generator(train_generator,
                               steps_per_epoch=2000 // BATCH_SIZE,
                               epochs=25,
                               validation_data=valid_generator,
                               validation_steps=200 // BATCH_SIZE)

    return hist

def cnn_appel_vgg(x_train, y_train, x_valid, y_valid, liste_train, liste_test):
    """
    Fonction de transfert learning
    """

    # On crée un modèle déjà pré entrainé
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(T_IMG, T_IMG, 3))

    # On rajoute les deux dernières couches qui nous intéressent
    x_model = base_model.output
    x_model = Flatten()(x_model)
    x_model = Dense(NB_RACES, activation='softmax')(x_model)

    # On crée notre modèle à partir de celui existant, et des deux couches en plus
    model = Model(inputs=base_model.input, outputs=x_model)

    # On choisi d'entrainer que nos couches rajoutées
    for layer in base_model.layers:
        layer.trainable = False

    # Compilatioon du modèle
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # On se donne des tours sans évolution pour stopper le fit
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc',
                                                    patience=5,
                                                    verbose=1)]

    # Visualisation de toutes les couches du modèle
    model.summary()

    if DATA_AUGMENTATION is True:
        res = cnn_data_augmentation(model, liste_train, liste_test)
    else:
        # Entrainement
        res = model.fit(x_train,
                        y_train,
                        epochs=25,
                        validation_data=(x_valid, y_valid),
                        verbose=1)

    # Dump (sauvegarde)
    joblib.dump(model, DOSSIER_SAVE + 'savefit.pkl')

    # Tracé de courbes pour visualiser les résultats
    cnn_courbes(res)

    return model

def cnn_courbes(resultat):
    """
    Tracé des courbes d'accuracy et de log loss
    """

    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(resultat.history['loss'], 'r', linewidth=3.0)
    plt.plot(resultat.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.tight_layout()
    plt.savefig(DOSSIER_SAVE + 'courbe_loss', dpi=100)
    plt.show()

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(resultat.history['acc'], 'r', linewidth=3.0)
    plt.plot(resultat.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.tight_layout()
    plt.savefig(DOSSIER_SAVE + 'courbe_accuracy', dpi=100)
    plt.show()

def gestion_erreur(res, test_y, labels, classifieur):
    """
    Gestion de l'erreur quand une catégorie de chien n'est pas prédite
    On rajoute la colonne vide manuellement
    """

    # Si ce n'est pas un kmeans, le traitement est différent (noms ou numéros)
    if classifieur == 'kmeans':
        for i in range(0, NB_RACES):
            if i not in res.columns:
                res[i] = 0

        for i in np.unique(labels):
            if i not in res.index:
                res.loc[i] = 0

    elif classifieur == 'cnn':
        for i in res.index:
            if i not in res.columns:
                res[i] = 0
    else:
        for i in np.unique(test_y):
            if i not in res.columns:
                res[i] = 0

        for i in res.columns:
            if i not in res.index:
                res.loc[i] = 0

    res = res.sort_index(axis=0, ascending=True)
    res = res.sort_index(axis=1, ascending=True)

    return res

def cnn_etablir_liste_chiens():
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
                    liste_images.append(filename)

                    # Rajout du label
                    labels.append(dirs[dirs.find('-')+1:].lower())

    liste_images = pd.DataFrame(liste_images, columns=['liste'])
    liste_images['labels'] = labels
    liste_images.to_csv('/home/toni/Bureau/liste.csv')

def cnn_recup_images():
    """
    Fonction de récupération des images
    """

    # Listes vides
    x_train = []
    y_train = []
    x_test = []

    # Etablissement du dataset de manière aléatoire
    cnn_etablir_liste_chiens()

    #liste_images.to_csv('/home/toni/Bureau/liste.csv')
    liste_images = pd.read_csv('/home/toni/Bureau/liste.csv')
    del liste_images['Unnamed: 0']

    # Séparation des datasets testing/training
    liste_train, liste_test = train_test_split(liste_images,
                                               test_size=0.2)

    liste_train = liste_train.reset_index(drop="True")
    liste_test = liste_test.reset_index(drop="True")

    # Préparation du one-hot encoding
    targets_series = pd.Series(liste_train['labels'])
    one_hot = pd.get_dummies(targets_series, sparse=True)
    one_hot_labels = np.asarray(one_hot)

    # Récupération des images et des labels de training
    i = 0
    for file, dump in tqdm(liste_train.values):
        # Lecture de l'image
        img = cv2.imread(DOSSIER_SOURCE + file)

        # Rajout des données dans la liste
        x_train.append(cv2.resize(img, (T_IMG, T_IMG)))

        # Race du chien
        y_train.append(one_hot_labels[i])
        i = i + 1

    # Récupération des images de testing
    for file in tqdm(liste_test['liste'].values):
        # Lecture de l'image
        img = cv2.imread(DOSSIER_SOURCE + file)

        # Rajout des données dans la liste
        x_test.append(cv2.resize(img, (T_IMG, T_IMG)))

    return liste_train, liste_test, x_train, y_train, x_test, one_hot

def main():
    """
    Fonction principale
    """

    # Debut du decompte du temps
    start_time = time.time()

    liste_train, liste_test, x_train, y_train, x_test, liste_noms = cnn_recup_images()

    # Reformatage des listes pour le format de VGG19
    y_train_raw = np.array(y_train, np.uint8)
    x_train_raw = np.array(x_train, np.float32) / 255.
    x_test = np.array(x_test, np.float32) / 255.
    print(x_train_raw.shape)
    print(y_train_raw.shape)
    print(x_test.shape)

    # Séparation des datasets training/validation
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_raw,
                                                          y_train_raw,
                                                          test_size=0.3,
                                                          random_state=1)

    # Appel de la fonction de transfer learning
    model = cnn_appel_vgg(x_train,
                          y_train,
                          x_valid,
                          y_valid,
                          liste_train,
                          liste_test)

    # Calcul des résultats
    predictions = model.predict(x_valid)
    predictions = np.argmax(predictions, axis=1)
    truth = np.argmax(y_valid, axis=1)

    res = pd.crosstab(np.asarray(truth),
                      predictions,
                      rownames=["Actual"],
                      colnames=["Predicted"])

    # Rajout des noms dans l'index et les colonnes
    res.index = liste_noms.columns
    res.columns = liste_noms.columns

    # Sauvegarde des classes
    liste_noms.to_csv(DOSSIER_SAVE + 'savefit.csv')

    # Gestion d'une erreur
    if len(res.columns) != NB_RACES:
        res = gestion_erreur(res, predictions, liste_train['labels'], 'cnn')
    cnn_calcul_resultats(res, np.asarray(predictions), 'cnn')

    errors = np.where(predictions != truth)[0]
    print("No of errors =", len(errors), "/", len(predictions))

#    # Visualisation des images mal labelisées
#    # Check Performance
#    fnames = test_generator.filenames
#    label2index = test_generator.class_indices
#    prob = model.predict(test_feats)
#
#    # Getting the mapping from class index to class label
#    idx2label = dict((v, k) for k, v in label2index.items())
#
#    for i in range(len(errors)):
#        pred_class = np.argmax(prob[errors[i]])
#        pred_label = idx2label[pred_class]
#
#        label_initial = np.argmax(test_labels[errors[i]])
#        label_initial = idx2label[label_initial]
#
#        print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#            label_initial,
#            pred_label,
#            prob[errors[i]][pred_class]))
#
#        original = load_img('{}/{}'.format(DOSSIER_SOURCE, fnames[errors[i]]))
#        plt.imshow(original)
#        plt.show()

    # Affichage du temps d execution
    print("Temps d execution :", round((time.time() - start_time), 2), 'secondes')
