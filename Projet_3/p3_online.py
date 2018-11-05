"""
Created on Wed Feb 14 20:22:34 2018

Version online du fichier de recherche de films.

@author: Toni
"""
# On importe les librairies dont on aura besoin pour ce tp
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from flask import Flask

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\'
_FICHIERDATA = _DOSSIER + 'p3_bdd_clean_v2.csv'
_FICHIERDATANUM = _DOSSIER + 'p3_datanum.csv'

app = Flask(__name__)

@app.route('/<id_film>')

def recommand(id_film):
    """
    Fonction d'entrée pour la recherche.
    """

    # Récupération des dataset
    data = pd.read_csv(_FICHIERDATA, encoding="ISO-8859-1")
    datanum = pd.read_csv(_FICHIERDATANUM, encoding="ISO-8859-1")

    # Suppresion des caractères de fin de lignes
    data = data.replace({'\xa0': ''}, regex=True)

    # Suppression des colonnes superflues
    del data['Unnamed: 0']
    del datanum['Unnamed: 0']

    return recommandation(datanum, data, id_film)

def recommandation(datanum, data, id_film):
    """
    Fonction qui calcule en fonction des dataset et du film d'entrée
    """

    # String initial vide
    texte_final = ''

    # Bloc try/exception pour voir si on a des résultats ou pas
    try:
        # Rajout des noms
        datanum['movie_title'] = data['movie_title']

        # Recherche des données du film
        data_film = datanum.loc[datanum['movie_title'].str.contains(id_film)]

        # Quel est le titre recherché
        mask = datanum['movie_title'].str.contains(id_film)
        titre = data['movie_title'][mask]

        # Astuce pour avoir un string au bon format
        for i in titre[-1:]:
            titre_fin = i

        # Suppression de la colonne non-chifrée
        del data_film['movie_title']
        del datanum['movie_title']

        # Valeur numérique si rien n'a été trouvé avec le "string"
        if data_film.empty:
            # Titre recherché
            titre_fin = data['movie_title'].loc[int(id_film)]

            # Valeurs binaires du film
            data_film = datanum.loc[int(id_film)].values.reshape(1, -1)

    except:
        # Rien n'a été trouvé
        texte_final = "Le film recherché n'existe pas."

    # Transformation en numpy array pour la suite du traitement
    data_film = np.array(data_film)

    # Si on a trouvé quelque chose, on continue le traitement
    # Sinon, on sort sans rien faire
    if data_film.size > 0:

        # Nom du film retenu
        texte_final = 'Titre retenu : ' + titre_fin + '<br>'

        # configuration du knn
        neigh = NearestNeighbors(n_neighbors=20,
                                 algorithm='auto',
                                 metric='euclidean'
                                )

        # knn
        neigh.fit(datanum)
        indices = neigh.kneighbors(data_film)

        # Récupération des 20 films
        for i in indices[1]:
            second_df = data.loc[i]

        # Liste qui va récupérer les noms de films à supprimer
        indice_supp = []

        # Recherche des films de mêmes séries
        for i in indices[1][-1]:
            if str(data.loc[i]['movie_title']) in titre_fin:
                indice_supp.append(i)
            elif titre_fin in str(data.loc[i]['movie_title']):
                indice_supp.append(i)

        # Suppression effective des films de même série
        #for i in indice_supp:
        #    second_df = second_df.drop([i])

        # Création du second dataset pour tester la popularité
        liste_criteres = ['movie_title',
                          'cast_total_facebook_likes',
                          'imdb_score',
                          'num_user_for_reviews',
                          'num_voted_users',
                          'movie_facebook_likes']

        # Suppresion des colonnes du dataset inutiles pour le calcul de popularité
        for colon in second_df:
            if colon not in liste_criteres:
                del second_df[colon]

        # On enlève les Nan
        second_df.fillna(0, inplace=True)

        # Tester la popularité
        reponse = popularite(second_df, id_film, indice_supp)
        texte_final = texte_final + reponse

    else:
        texte_final = "Le film recherché n'existe pas."

    return texte_final

def popularite(second_df, id_film, indice_supp):
    """
    Calcul de la popularité des 20 films pré-selectionnés
    """

    # Scale des données (division par la valeur maximum)
    min_max_scaler = preprocessing.MinMaxScaler()
    second_df[['cast_total_facebook_likes',
               'imdb_score',
               'num_user_for_reviews',
               'num_voted_users',
               'movie_facebook_likes']] = min_max_scaler.fit_transform(second_df[['cast_total_facebook_likes', 'imdb_score', 'num_user_for_reviews', 'num_voted_users', 'movie_facebook_likes']])

    # Indice de popularité simpliste
    second_df['score'] = second_df['cast_total_facebook_likes'] + second_df['imdb_score'] + second_df['movie_facebook_likes'] + second_df['num_user_for_reviews'] + second_df['num_voted_users']

    # Score du film
    score = sum(second_df.score[second_df.movie_title == id_film])

    # Suppression effective des films de même série
    for i in indice_supp:
        second_df = second_df.drop([i])
            
    # Calcul de la valeur absolue du score pour voir la différence avec les autres films
    second_df['scoreP'] = abs(second_df['score']-score)
    second_df = second_df.sort_values(by='scoreP', ascending=True)

    # On ne garde que les 5 résultats les plus proches sans prendre le film lui-même
    second_df = second_df[0:5]

    # Transformation en dictionnaire
    dico = {"_results" :[{'id':int(key), "name":value} for key, value in second_df['movie_title'].items()]}

    # Transformation en format JSON
    json_dico = json.dumps(dico, indent=4, separators=(',', ': '))

    # On renvoie le résultat en format JSON
    return json_dico
