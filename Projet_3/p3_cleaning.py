# -*- coding: utf-8 -*-

# P3 : Data cleaning

# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm as cm
from sklearn import linear_model
#from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute

# Lieu où se trouve le fichier
_FICHIER = 'C:\\Users\\Toni\\Desktop\\movie_metadata.csv'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_3\\images'

# function that extract statistical parameters from a grouby objet
def get_stats(param):
    """
    TBD
    """
    return {'min':param.min(),
            'max':param.max(),
            'count': param.count(),
            'mean':param.mean()
           }

def count_word(data, ref_col, liste):
    """
    TBD
    """
    keyword_count = dict()

    for word in liste:
        keyword_count[word] = 0

    for liste_keywords in data[ref_col].str.split('|'):
        if isinstance(liste_keywords, float) and pd.isnull(liste_keywords):
            continue
        for word in [word for word in liste_keywords if word in liste]:
            if pd.notnull(word):
                keyword_count[word] = keyword_count[word] + 1

    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []

    for k, v in keyword_count.items():
        keyword_occurences.append([k, v])

    keyword_occurences.sort(key=lambda x: x[1], reverse=True)

    return keyword_occurences, keyword_count

def afficher_plot(type_donnee, trunc_occurences):
    """
    TBD
    """
    fichier_save = _DOSSIERTRAVAIL + '\\' + type_donnee

    words = dict()

    for word in trunc_occurences:
        words[word[0]] = word[1]

    plt.figure(figsize=(15, 10))
    y_axis = [i[1] for i in trunc_occurences]
    x_axis = [k for k, i in enumerate(trunc_occurences)]
    x_label = [i[0] for i in trunc_occurences]
    plt.xticks(rotation=90, fontsize=10)
    plt.xticks(x_axis, x_label)

    plt.yticks(fontsize=10)
    plt.ylabel("Nb. of occurences", fontsize=18, labelpad=10)

    plt.bar(x_axis, y_axis, align='center', color='b')

    plt.savefig(fichier_save, dpi=100)

    plt.title(type_donnee + " popularity", fontsize=25)
    plt.show()

def comptabiliser(data, valeur_cherchee):
    """
    TBD
    """
    # compter tous les genres différents
    listing = set()

    for word in data[valeur_cherchee].str.split('|').values:
        if isinstance(word, float):
            continue
        listing = listing.union(word)

    # compter le nombre d'occurence de ces genres
    listing_compte, dum = count_word(data, valeur_cherchee, listing)

    return listing_compte

def correlation_matrix(data):
    """
        Fonction qui permet de créer une visualisation du lien entre les
        variables 2 à 2
    """
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
    plt.show()

def histogramme(data, colon):
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
    # plt.hist(data[colon], bins=bin_values)
    # Test sans les valeurs NaN
    plt.hist(data[colon][np.isfinite(data[colon])], bins=100)
    plt.savefig(fichier_save, dpi=100)

def scatter_plot(data, nom_colonne2, nom_colonne):
    """
        Fonction qui permet d'afficher les nuages de points
    """

    #Log
    print("Fct affichage_plot\n")

    data = data[data[nom_colonne] <= data[nom_colonne].quantile(0.98)]
    data = data[data[nom_colonne2] <= data[nom_colonne2].quantile(0.98)]

    # Déliminations du visuel pour x
    xmax = max(data[nom_colonne])
    ymax = max(data[nom_colonne2])

    # Déliminations du visuel pour y
    xmin = min(data[nom_colonne])
    ymin = min(data[nom_colonne2])

    # création du nuage de point avec toujours la même ordonnée
    data.plot(kind="scatter", x=nom_colonne, y=nom_colonne2)

    # Affichage
    plt.grid(True)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()

def lasso(data):

    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn import preprocessing

    # On charge le dataset
    data = pd.read_csv(_FICHIER)

    # Suppresion de doublons
    data=data.drop_duplicates(subset = ['movie_title', 'actor_1_name', 'director_name'], keep='first')

    data = data.drop(['color', 'director_name', 'actor_1_name', 'genres', 'movie_title', 'actor_2_name', 'actor_3_name'], axis=1)
    data = data.drop(['plot_keywords', 'movie_imdb_link', 'language', 'country', 'content_rating'], axis=1)

    data.fillna(-1, inplace=True)

    X = data.copy()
    del X['gross']
    y = data['gross']
    
    std_scale = preprocessing.StandardScaler().fit(X)
    X_scaled = std_scale.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2)

    # On crée un modèle de régression linéaire
    lr = linear_model.LinearRegression()

    # On entraîne ce modèle sur les données d'entrainement
    lr.fit(X_train, y_train)

    # On récupère l'erreur de norme 2 sur le jeu de données test comme baseline
    baseline_error = np.mean((lr.predict(X_test) - y_test) ** 2)

    print(baseline_error)

    n_alphas = 500
    alphas = np.logspace(-10, 10, n_alphas)
    ridge = linear_model.Ridge()
    
    coefs = []
    errors = []
    for a in alphas:
        ridge.set_params(alpha=a)
        ridge.fit(X_train, y_train)
        coefs.append(ridge.coef_)
        errors.append([baseline_error, np.mean((ridge.predict(X_test) - y_test) ** 2)])
    
    ax = plt.gca()
    
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.axis('tight')
    plt.show()

    ax = plt.gca()
    
    ax.plot(alphas, errors)
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('error')
    plt.axis('tight')
    plt.show()

    errors[np.argmin(errors)][1]
    alphas[np.argmin(errors)]
    
def input_reg_linear(data, colon_predict, colon_ref):
    """
    TBD
    """

    # Création de la régression linéaire
    regr = linear_model.LinearRegression()
    test = data[[colon_predict, colon_ref]].dropna(how='any', axis=0)

    # Conversion en NP
    x_data = np.array(test[colon_ref])
    y_data = np.array(test[colon_predict])

    # Reshape obligatoire
    x_data = x_data.reshape(len(x_data), 1)
    y_data = y_data.reshape(len(y_data), 1)

    # On fit les données préparées
    regr.fit(x_data, y_data)

    test = data[data[colon_predict].isnull() & data[colon_ref].notnull()]

    for index, row in test.iterrows():
        value = float(regr.predict(row[colon_ref]))
        data.set_value(index, colon_predict, value)

def replace_classification(data):
    
    data['content_rating'] = data['content_rating'].replace('Unrated', 'Not Rated')
    data['content_rating'] = data['content_rating'].replace('Approved', 'Passed')
    data['content_rating'] = data['content_rating'].replace('TV-14', 'R')
    data['content_rating'] = data['content_rating'].replace('TV-MA', 'NC-17')
    data['content_rating'] = data['content_rating'].replace('TV-PG', 'PG-13')
    data['content_rating'] = data['content_rating'].replace('GP', 'PG')
    data['content_rating'] = data['content_rating'].replace('X', 'NC-17')
    data['content_rating'] = data['content_rating'].replace('M', 'R')
    data['content_rating'] = data['content_rating'].replace('TV-Y', 'G')
    data['content_rating'] = data['content_rating'].replace('TV-Y7', 'G')
    data['content_rating'] = data['content_rating'].replace('TV-G', 'PG')
    data['content_rating'] = data['content_rating'].replace('Passed', 'G')
    
    data['content_rating'] = data['content_rating'].replace('G', 'G-')
    data['content_rating'] = data['content_rating'].replace('PG-13', 'PG13')
    data['content_rating'] = data['content_rating'].replace('PG', 'P-G')
    data['content_rating'] = data['content_rating'].replace('R', 'Rated-R')
    
def main():
    """
    TBD
    """
    # On charge le dataset
    data = pd.read_csv(_FICHIER)

    # Suppresion de doublons
    data=data.drop_duplicates(subset = ['movie_title', 'actor_1_name', 'director_name'], keep='first')

    # Données manquantes
    print("Données manquantes")
    missing_data = data.isnull().sum(axis=0).reset_index()

    # Change les noms des colonnes
    missing_data.columns = ['column_name', 'missing_count']

    # Crée une nouvelle colonne et fais le calcul des données manquantes
    missing_data['filling_factor'] = (data.shape[0]-missing_data['missing_count'])/data.shape[0]*100

    # Classe et affiche
    missing_data.sort_values('filling_factor').reset_index(drop=True)

    # Transposition du dataframe de données pour l'analyse univariée
    fichier_save = _DOSSIERTRAVAIL + '\\' + 'transposition.csv'
    data_transpose = data.describe().reset_index().transpose()
    print(data_transpose)
    data_transpose.to_csv(fichier_save)

    # Matrice de correlation
    # The movie "gross" has strong positive correlation with the "num_voted_users"
    correlation_matrix(data)

    data_reg = data.copy()

    # Masque pour virer les valeurs NaN
    # mask_colon1 = ~np.isnan(data_reg['gross'])
    mask_colon1 = np.isfinite(data_reg['gross'])

    #data_reg.fillna(0, inplace=True)

    colon1 = 'gross'

    for colon2 in data:
        if (data_reg[colon2].dtype == 'float' or data_reg[colon2].dtype == 'int64') and colon2 != colon1:

            # mask_colon2 = ~np.isnan(data_reg[colon2])
            mask_colon2 = np.isfinite(data_reg[colon2])
            mask = mask_colon1 & mask_colon2

            # Calcul d'une regression linéaire
            regr = linear_model.LinearRegression()

            # Reshape
            data_x = data_reg[colon1].values.reshape(-1, 1)
            data_y = data_reg[colon2].values.reshape(-1, 1)

            # Fit
            regr.fit(data_x[mask], data_y[mask])

            # Affichage de la variances : On doit être le plus proche possible de 1
            print('Regression sur :', colon1, colon2)
            print('Score : %.2f' % np.corrcoef(data_reg[colon1][mask], data_reg[colon2][mask])[1, 0])
            print('R2    : %.2f \n' % regr.score(data_x[mask], data_y[mask]))

    # Affichage des nuages de points
    for nom_colonne in data:
        if data[nom_colonne].dtype == 'float' or data[nom_colonne].dtype == 'int64':
            scatter_plot(data, 'gross', nom_colonne)

    # Création des histogrammes
    for nom_colonne in data:
        if data[nom_colonne].dtype == 'float' or data[nom_colonne].dtype == 'int64':
            histogramme(data, nom_colonne)

    # Partie pour le recensement
    # Création de la database avec tous les noms d'acteurs car ils sont sur
    # 3 colonnes différentes
    db_names = []
    db_names.extend(data['actor_1_name'])
    db_names.extend(data['actor_2_name'])
    db_names.extend(data['actor_3_name'])
    data_names = pd.DataFrame(db_names, columns=['name'])

    # compter tous les acteurs de films
    actors_list = comptabiliser(data_names, 'name')
    afficher_plot('actors', actors_list[0:50])

    list_a_afficher = ['genres', 'language', 'country', 'content_rating', 'plot_keywords', 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']

    replace_classification(data)

    for name in list_a_afficher:
        res = comptabiliser(data, name)
        afficher_plot(name, res[0:50])

    colon_predict = 'gross'
    colon_ref = 'num_voted_users'
    input_reg_linear(data, colon_predict, colon_ref)

    data.to_csv('C:\\Users\\Toni\\Desktop\\pas_synchro\\p3_bdd_clean_v2.csv')
