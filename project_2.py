import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer, KNNImputer

st.image('pellicule_film.jpg',width=600)


netflix = pd.read_csv('final_movies_DB_2.csv', header=0, sep=',', quotechar='"', error_bad_lines=False)



netflix = netflix[(netflix["runtimeMinutes"] >= 80) &  



                  (netflix["runtimeMinutes"] <= 240) & 



                  (netflix["numVotes"] >= 15000) & (netflix["averageRating"] >= 5) & 



                  (netflix["startYear"] >= 1950) & (netflix["startYear"] <= 2021)] 




netflix_liste = netflix[['originalTitle', 'title', 'startYear', 'runtimeMinutes', 'averageRating', 'numVotes', 'Main_genre', 'Directors', 'Actor_1']]



top_films = netflix_liste.sort_values(['averageRating','numVotes'], ascending=(False,False))



for_show = top_films.head()



st.write("Voici les films les mieux notés (sur la base imdb) :")



st.write(for_show.assign(hack='').set_index('hack'))

# On reduit la base pour permettre à l'algorithm une rapidité de calcul optimal
netflix = netflix[(netflix["runtimeMinutes"] >= 80) &  
                  (netflix["runtimeMinutes"] <= 240) & 
                  (netflix["numVotes"] >= 15000) & (netflix["averageRating"] >= 5) & 
                  (netflix["startYear"] >= 1950) & (netflix["startYear"] <= 2021)] 

# On normalise les données des différentes variables utilisées pour calculer le knn
scaled_features = netflix.copy()
col_names = ['numVotes', 'averageRating', 'runtimeMinutes']
features = scaled_features[col_names]
scaler = RobustScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features[col_names] = features

# On définit toutes les  variables que nous utiliseront pour l'algo
scaled_features = scaled_features[["title","originalTitle","startYear","runtimeMinutes","averageRating","Main_genre","Sub_genre_1","Sub_genre_2"]]

# On normalise les variables categorielles en utilisant get_dummies
scaled_features = pd.get_dummies(scaled_features, columns =["Main_genre","Sub_genre_1","Sub_genre_2"])
scaled_features.dropna(inplace=True)

# On définit nos variables X et y
X = scaled_features.drop(["originalTitle","title","startYear"], axis=1)
y = scaled_features["originalTitle"]

# On choisit le nombre de films que nous souhaitons proposer
knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn.fit(X, y)

# On récupère les indices des films les plus proches
distances, indices = knn.kneighbors(X)

netflix_liste_vf = netflix['title'].tolist()
netflix_liste_vo = netflix['originalTitle'].tolist()
netflix_liste = netflix_liste_vf + netflix_liste_vo
netflix_liste.insert(0, '')

st.write('Si vous aimez ... (Choississez un film) :')
film_select = st.selectbox('', netflix_liste)

film_select = scaled_features[np.where(scaled_features['originalTitle'].str.contains(film_select,case=False), True,False)|
               np.where(scaled_features['title'].str.contains(film_select,case=False), True,False)]

film_select_string = film_select.iloc[0,0]

propositions = knn.kneighbors(scaled_features.loc[scaled_features['title'] == film_select_string, 'runtimeMinutes':'Sub_genre_2_Romance'])

final_proposition = propositions[1][0]
final_proposition = final_proposition.tolist()

prop = netflix.iloc[final_proposition]
prop = prop[['title', 'originalTitle', 'startYear', 'runtimeMinutes', 'averageRating', 'numVotes', 'Main_genre', 'Sub_genre_1', 'Directors']]
prop = prop.iloc[1:4,:]

if st.checkbox(''):
    st.write('... vous aimerez certainement : ', prop.assign(hack='').set_index('hack'))






