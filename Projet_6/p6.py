# -*- coding: utf-8 -*-
"""
Created on Thu May 17 21:51:03 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import sys
import warnings
import collections
import pandas as pd

from bs4 import BeautifulSoup
from nltk import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from mosestokenizer import MosesDetokenizer
from sklearn.externals import joblib
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from yellowbrick.text import FreqDistVisualizer

# Pour ne pas avoir les warnings lors de la compilation
warnings.filterwarnings("ignore")

_RECALCUL_JOBLIB = False
_MINDF = 0.01
_N_COMPONENTS = 20

# Lieu où se trouve le fichier
if sys.platform == "linux":
    _FICHIER = 'stackoverflow_train_dataset.csv'
    _DOSSIER = '/home/toni/Bureau/'
    _DOSSIERTRAVAIL = '/home/toni/python/Projet_6/images/'
    _DOSSIERPKL = '/home/toni/python/Projet_6/pkl/'

else:
    _FICHIER = 'QueryResults.csv'
    _DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p6\\'
    _DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_6\\images\\'
    _DOSSIERPKL = 'C:\\Users\\Toni\\python\\python\\Projet_6\\pkl\\'

def exploration(data):
    """
    Phase d'exploration des données
    """

    # Affichae du score des questions
    score_per_question = collections.Counter(data['Score'])
    scorewerid, noanswers = zip(*score_per_question.most_common())

    for number in [10, 25, 50]:
        plt.bar(scorewerid[:number], noanswers[:number], align='center', alpha=0.75)
        titre = 'Score des questions pour N=' + str(number)
        plt.ylabel('Nombre de questions')
        plt.xlabel('Score')
        plt.title(titre)
        plt.savefig(_DOSSIERTRAVAIL + titre, dpi=100)
        plt.show()

    # Tags plus fréquents
    tagcount = comptage(list(data['Tags'])).most_common(10)
    print(tagcount)
    axe_x, axe_y = zip(*tagcount)

    plt.figure(figsize=(9, 8))
    titre = 'Ocurrence des tags'
    plt.title(titre)
    plt.ylabel("Nombre de questions concernées")
    for i in range(len(axe_y)):
        plt.bar(i, axe_y[i], align='center', alpha=0.75, label=axe_x[i])

    plt.legend(numpoints=1)
    plt.savefig(_DOSSIERTRAVAIL + titre, dpi=100)
    plt.show()

def creer_tfidfvectorizer(text):
    """
    Fonction de création de la matrice numérique
    Comptage de fréquence
    """

    # Création de l'objet
    t_vectorizer = TfidfVectorizer(min_df=_MINDF)

    # Fit du texte d'entrée, et mis au format tableau
    liste_mots = t_vectorizer.fit_transform(text).toarray()

    # On ressort le tableau, et la liste des mots
    return liste_mots, t_vectorizer.get_feature_names()

def fct_nltk(text, stop_words):
    """
    Fonction pour supprimer :
        les step words
        la ponctuation
        les majuscules
        les pluriels
    """

    # Création de l'objet
    lemma = wordnet.WordNetLemmatizer()

    # Tokenization et mise en minuscule
    words = word_tokenize(text.lower())

    # Suppression des pluriels et de la ponctuation. Boule pour toutes les lignes
    new_sentence = [lemma.lemmatize(x) for x in words if (not x in stop_words) and x.isalpha()]

    # Sortie
    return new_sentence

def suppr_html_code(fichier):
    """
    Fonction de suppression des balises HTML et des parties de code
    """

    # Suppression des balises HTML
    soupe = BeautifulSoup(fichier, "lxml")

    # Recherche des balises de 'code'
    liste = soupe.findAll('code')

    # Suppression des données qui sont entre les balises de code
    for balise in liste:
        balise.decompose()

    # Sortie formatée en texte
    return soupe.text

def donnees_manquantes(data, nom):
    """
    Données manquantes
    """

    # Données manquantes
    fichier_save = _DOSSIERTRAVAIL + nom + '.csv'
    missing_data = data.isnull().sum(axis=0).reset_index()
    missing_data.columns = ['column_name', 'missing_count']
    missing_data['filling_factor'] = (data.shape[0]-missing_data['missing_count'])/data.shape[0]*100
    print(missing_data.sort_values('filling_factor').reset_index(drop=True))
    missing_data.sort_values('filling_factor').reset_index(drop=True).to_csv(fichier_save)

    # Transposition du dataframe de données pour l'analyse univariée
    fichier_save = _DOSSIERTRAVAIL  + 'transposition.csv'
    data_transpose = data.describe().reset_index().transpose()
    print(data_transpose)
    data_transpose.to_csv(fichier_save)

def display_topics(model, feature_names, no_top_words, vectype):
    """
    Affichage des topics et d'un wordcloud
    """

    # Pour tous les topics envoyées pas le model
    for idx, topic in enumerate(model.components_):
        print("\nTopic", idx)
        listing = []

        # On rajoute chaque mot dans une liste qu'on va afficher
        for i in topic.argsort()[:-no_top_words - 1:-1]:
            listing.append(feature_names[i])

        # On affiche la liste
        print(listing)

        # Appel de la fonction pour générer le nuage de mot
        generate_wordcloud(listing, idx, vectype)

def reduction_dimension(data, limit, vectype):
    """
    Fonction de réduction de dimension
    """

    # TruncatedSVD pour réduction de dimension
    svd = TruncatedSVD(n_components=limit,
                       n_iter=5)

    svd.fit(data)

    #print(svd.explained_variance_ratio_)
    #print(svd.singular_values_)
    print("Pour", limit, "composants :", round(100*svd.explained_variance_ratio_.sum(), 2), "%")

    # Définition des axes x et y
    abs_x = range(0, limit)
    ord_y = [svd.explained_variance_ratio_[0:i].sum() for i in range(0, limit)]

    # Affichage
    plt.plot(abs_x, ord_y)
    title = 'Reduction de dimension pour ' + vectype
    plt.title(title)
    plt.tight_layout()
    plt.savefig(_DOSSIERTRAVAIL + vectype + '_reduction_', dpi=100)
    plt.show()

    # On garde la dernière valeur de i dans le fit du dessus ?
    data_reduit = svd.fit(data).transform(data)

    return data_reduit

def generate_wordcloud(data, cpt, vectype):
    """
    Simple WordCloud
    """

    # Génération du wordcloud
    wordcloud = WordCloud(background_color='black',
                          #stopwords=stopwords,
                          max_words=20,
                          max_font_size=20,
                          relative_scaling=1,
                          min_font_size=1,
                          scale=3,
                          random_state=1
                         ).generate(str(data))

    # Affichage
    fig = plt.figure(1, figsize=(6, 6))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(wordcloud)
    plt.savefig(_DOSSIERTRAVAIL + vectype + '_wordcloud_' + str(cpt), dpi=100)
    plt.show()

def comptage(data):
    """
    Fonction qui va compter l'occurence de tous les mots
    """

    count = collections.Counter()

    stop_words = set(stopwords.words('english'))

    for sentence in data:
        sentence = word_tokenize(sentence.lower())
        for word in sentence:
            if (not word in stop_words) and word.isalpha():
                count[word] += 1

    return count

def tfidf(new_df):
    """
    Création de la matrice tfidf
    """

    # Création de la matrice finale
    matrixnum_tfidf, names_tfidf = creer_tfidfvectorizer(new_df['Sentences'])
    # Conversion de la matrice finale en dataframe pour facilité d'usage
    matrixnum_tfidf = pd.DataFrame(matrixnum_tfidf,
                                   columns=names_tfidf,
                                   index=new_df.index)

    # Sépération des datasets
    matrixnum_tfidf_train, matrixnum_tfidf_test = train_test_split(matrixnum_tfidf,
                                                                   test_size=0.25,
                                                                   random_state=3)

    # Run LDA
    lda = LatentDirichletAllocation(n_components=_N_COMPONENTS,
                                    #max_iter=5,
                                    #learning_method='online',
                                    #learning_offset=50,
                                    random_state=0)

    # Fit du LDA crée au-dessus
    lda.fit(matrixnum_tfidf_train)

    # Visualisation de la liste des mots, plus nuage de mots
    #display_topics(lda, names_tfidf, 15, 'lda')

    # Visualisation de la fréquence d'occurence
    ## voir la fréquence minimale
    #visualizer = FreqDistVisualizer(features=names_tfidf,
    #                                n=20,
    #                                orient='h',
    #                                color='g')
    #visualizer.fit(matrixnum_tfidf_train)
    #visualizer.poof()

    # Tentative de réduction de la dimension
    #matrixnum_tfidf_num = reduction_dimension(matrixnum_tfidf, 500, 'lda')

    return matrixnum_tfidf_train, matrixnum_tfidf_test, names_tfidf, lda

def comptage_metric(data, df_prevision, value):
    """
    Fonction de dénombrement
    """

    # Comptage des bons tags prédits
    count_tag = 0

    for i in df_prevision.index:
        liste_tags = word_tokenize(data.loc[i, 'Tags'])

        for tag in liste_tags:
            for j in range(0, value):
                if tag == df_prevision.loc[i, j]:
                    count_tag = count_tag + 1

    print("Bons tags / Nb de tags prédit")
    print(round((count_tag/df_prevision.count().sum())*100, 1), '%')

def non_supervise(new_df, data, liste_tags):
    """
    Il faut deux matrices (distribution de proba) : documents/topic et topic/mots
    puis multiplication des deux matrices
    """

    ### ESSAI AVEC TFIDFVECTORIZER
    matrixnum_tfidf_train, matrixnum_tfidf_test, names_tfidf, lda_tfidf = tfidf(new_df)

    # Probabilité d'appartanence d'une message à un topic
    df_tp1 = pd.DataFrame(lda_tfidf.transform(matrixnum_tfidf_test))
    df_tp1.index = matrixnum_tfidf_test.index

    ## PARTIE 1
    # Probabilité d'appartanence d'un mot à un topic
    df_tp3 = lda_tfidf.components_ #/ lda_tfidf.components_.sum(axis=1)[:, np.newaxis]

    # Multiplication des deux matrices pour obtenir la proba documents/mots
    df_mots = df_tp1.dot(df_tp3)
    df_mots.columns = names_tfidf

    # Création de la matrice des mots les plus fréquents par document
    df_plus_frequent = pd.DataFrame()

    for i in df_mots.index:
        temp = df_mots.loc[i].nlargest(5)
        temp = temp.reset_index()
        df_plus_frequent[i] = temp['index']

    df_plus_frequent = df_plus_frequent.T
    df_plus_frequent.index = df_tp1.index

    # Comptage des bons tags prédits
    print("Avec mots")
    comptage_metric(data, df_plus_frequent, 5)

    ## PARTIE 2
    # Probabilité d'appartanence d'un tag à un topic
    df_tp2 = pd.DataFrame(columns=liste_tags, index=range(0, _N_COMPONENTS))

    # Comptage d'occurence d'apparition des tags pour chaque topic
    for i in liste_tags:
        mot = ' ' + i + ' '
        mask = data['Tags'].str.contains(str(mot), regex=False)
        temp = df_tp1[mask]

        for j in df_tp1.columns:
            df_tp2.loc[j, i] = temp[j].sum()

    # Convertion en float
    df_tp2 = df_tp2.astype(float)

    # Multiplication des deux matrices pour obtenir la proba documents/mots
    df_tags = df_tp1.dot(df_tp2)

    # Transformation en dataframe
    df_tags = df_tags.astype(float)

    # Création de la matrice pour afficher les mots les plus fréquents par document
    df_prevision = pd.DataFrame()

    for i in df_tags.index:
        temp = df_tags.loc[i].nlargest(5)
        temp = temp.reset_index()

        # On supprime ceux qui ne sont pas dans le body de la question
        for k, j in enumerate(temp['index']):
            mot = ' ' + j + ' '
            if mot not in data.loc[i, 'Body']:
                temp = temp.drop(k)

        df_prevision = df_prevision.append(temp['index'])

    df_prevision = pd.DataFrame(df_prevision).reset_index(drop=True)
    df_prevision.index = df_tp1.index

    # Comptage des bons tags prédits
    print("Avec tags")
    comptage_metric(data, df_prevision, 5)

    return matrixnum_tfidf_train, matrixnum_tfidf_test

def supervise(data, matrixnum_tfidf_train, matrixnum_tfidf_test, df_tags_train, df_tags_test):
    """
    # one hot encoding sur un set de tag assez fréquent
    # one versus rest classifier
    # sklearn multi label
    """

    # Choix de l'algorithme de classification
    models = [RandomForestClassifier(n_estimators=10, max_depth=30),
              RandomForestClassifier(n_estimators=10, max_depth=None),
              RandomForestClassifier(n_estimators=30, max_depth=30),
              RandomForestClassifier(n_estimators=30, max_depth=None)]

    #models = [SGDClassifier(loss='log', penalty='none'),
    #          SGDClassifier(loss='log', penalty='l2'),
    #          SGDClassifier(loss='log', penalty='l1'),
    #          SGDClassifier(loss='log', penalty='elasticnet'),
    #        ]

    for model in models:
        # Nom du classifieur
        #name = model.__class__.__name__ + '_' + str(model.max_depth) + '_' + str(model.n_estimators)
        name = model.__class__.__name__ + '_' + str(model.loss) + '_' + str(model.penalty)

        # Localisation de du fichier du fit sauvegardé
        fichier = _DOSSIERPKL + "/" + name + ".pkl"

        print(name, "\n")

        # Choix de l'algorithme de décision
        classif = OneVsRestClassifier(model)

        # Choix entre fit de nouveau ou aller chercher le fit sauvegardé
        if _RECALCUL_JOBLIB is True:
            # Fit
            classif.fit(matrixnum_tfidf_train, df_tags_train)
            # Dump (sauvegarde)
            joblib.dump(classif, fichier)
        else:
            # On va chercher le dump
            classif = joblib.load(fichier)

        # On va chercher les index sauvegardés pour gagner du temps sur la prédiction
        fichier3 = _DOSSIERTRAVAIL + "index.pkl"
        index_c = joblib.load(fichier3)

        liste_tags = matrixnum_tfidf_test.columns

        # Avec ces deux boucles, on les copies et on s'assure que tous les index
        # sont présents
        for i in index_c:
            if i not in liste_tags:
                matrixnum_tfidf_test[i] = 0

        for i in matrixnum_tfidf_test.columns:
            if i not in index_c:
                del matrixnum_tfidf_test[i]

        # Predictions
        predictions = classif.predict_proba(matrixnum_tfidf_test)
        predictions = pd.DataFrame(predictions,
                                   index=df_tags_test.index,
                                   columns=df_tags_test.columns)

        df_prevision_finale = pd.DataFrame()

        for pred in predictions.index:
            temp = predictions.loc[pred]
            temp = temp[temp > 0]
            temp = temp.nlargest(5)
            temp = temp.reset_index()

            # On supprime ceux qui ne sont pas dans le body de la question
            for k, val in enumerate(temp['index']):
                mot = ' ' + val + ' '
                if mot not in data.loc[pred, 'Body']:
                    temp = temp.drop(k)

            df_prevision_finale = df_prevision_finale.append(temp['index'])

        df_prevision_finale = pd.DataFrame(df_prevision_finale).reset_index(drop=True)
        df_prevision_finale.index = matrixnum_tfidf_test.index

        # Comptage des bons tags prédits
        print("Supervisé")
        comptage_metric(data, df_prevision_finale, 5)

        for k in df_prevision_finale.index[:20]:

            # On prends les lignes une par une
            ligne = df_prevision_finale.loc[k].copy()

            # Variable list qui va prendre les résultats des prédictions
            predicted_label = []

            for word in ligne:
                if not pd.isna(word):
                    predicted_label.append(word)

            # Impression du resultat
            if not predicted_label:
                print(data['Body'].loc[k][:80], "...")
                print('Actual label :', data['Tags'].loc[k])
                print("Predicted label :", predicted_label, "\n")

def main():
    """
    Fonction principale
    """

    # Récupération du dataset
    data = pd.read_csv(_DOSSIER + _FICHIER, error_bad_lines=False)

    # Reduction de la taille
    data = data[0:20000]

    # Exploration
    exploration(data)

    # Fusion du body et du title
    data['Body'] = data['Title'] + data['Body']
    data['Tags'] = data['Tags'].str.replace("<", "")
    data['Tags'] = data['Tags'].str.replace(">", " ")

    # Création de la liste des tags d'origines, uniques
    liste_tags = []

    for i in range(0, len(data)):
        words = word_tokenize(data.loc[i, 'Tags'])
        for j in words:
            if j.isalpha() and (j not in liste_tags):
                liste_tags.append(j)

    # Nouveau dataframe qui prendra les résultats en entrée
    new_df = pd.DataFrame()

    # Données manquantes
    donnees_manquantes(data, "missing_data_1")

    # Suppression des balises html et des parties de code
    data['Body'] = [suppr_html_code(x) for x in data['Body']]

    # Comptage du nombre d'occurence
    cpt = comptage(data['Body'])

    # Liste des stop words anglais
    least_used = set([word for word in cpt if cpt[word] < 100])

    stop_words = set(stopwords.words('english')) \
                | set([word for word, freq, in cpt.most_common(100)]) \
                | least_used

    # Suppression des pluriels et des stop words
    # Rétrecissment du dataset
    new_df['Sentences'] = [fct_nltk(x, stop_words) for x in data['Body']]

    # On est obligé de detokenizer pour créer la matrice finale
    detokenizer = MosesDetokenizer()
    new_df['Sentences'] = [detokenizer(x) for x in new_df['Sentences']]
    new_df['Tags'] = data['Tags']

    ### NON-SUPERVISE
    for i in [0.01, 0.03, 0.05]:
        _MINDF = i
        for j in [10, 20]:
            _N_COMPONENTS = j
            print(_MINDF, _N_COMPONENTS)
            matrixnum_tfidf_train, matrixnum_tfidf_test = non_supervise(new_df, data, liste_tags)

    # Sépération des datasets
    data_train, data_test = train_test_split(new_df,
                                             test_size=0.25,
                                             random_state=3)

    # One hot encoding en prenant en compte la séparation des tags
    df_tags = data['Tags'].str.get_dummies(sep=' ')
    df_tags_train = data_train['Tags'].str.get_dummies(sep=' ')
    df_tags_test = data_test['Tags'].str.get_dummies(sep=' ')

    # Réduction de la dimension de la matrice des tags,
    # on ne prends que les plus fréquents
    mask = pd.Series(df_tags.sum() > 50)
    df_tags_train = df_tags_train.loc[:, mask]
    df_tags_test = df_tags_test.loc[:, mask]

    ### SUPERVISE
    supervise(data, matrixnum_tfidf_train, matrixnum_tfidf_test, df_tags_train, df_tags_test)
