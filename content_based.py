import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load data
data_path = 'imdb.csv'
movies = pd.read_csv(data_path)

# Properly reassign DataFrame after dropping duplicates and missing values
movies = movies.drop_duplicates(subset='Series_Title', keep='first')
movies = movies.dropna(subset=['Overview', 'Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4'])

# Drop useless columns and reassigned
movies = movies.drop(columns=['Poster_Link'])

# Feature Engineering
# Text vectorization for 'Overview'
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['Overview'])

# Concatenate all text data that will be used to compute similarity
movies['all_text'] = movies['Genre'] + ' ' + movies['Director'] + ' ' + \
                     movies['Star1'] + ' ' + movies['Star2'] + ' ' + \
                     movies['Star3'] + ' ' + movies['Star4']
text_matrix = tfidf.fit_transform(movies['all_text'])
# Normalize numerical features after ensuring no NaNs are left
scaler = MinMaxScaler()
movies[['IMDB_Rating', 'Meta_score', 'No_of_Votes']] = scaler.fit_transform(
    movies[['IMDB_Rating', 'Meta_score', 'No_of_Votes']].fillna(0)  # Fill NaNs with 0 or handle appropriately
)

# Combine all features into a single similarity matrix
combined_features = np.hstack([text_matrix.toarray(), tfidf_matrix.toarray(),
                               movies[['IMDB_Rating', 'Meta_score', 'No_of_Votes']].values])
# Check for NaNs before computing similarity
if np.isnan(combined_features).any():
    print("NaN values detected in combined features.")
else:
    similarity_matrix = cosine_similarity(combined_features)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_recommendations(title, movies, similarity_matrix):
    # Get the index of the movie that matches the title
    idx = movies.index[movies['Series_Title'] == title].tolist()[0]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]  # Skip the first entry because it's the movie itself

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies['Series_Title'].iloc[movie_indices]

recommended_movies = get_recommendations('Monsters, Inc.', movies, similarity_matrix)
print(recommended_movies)

