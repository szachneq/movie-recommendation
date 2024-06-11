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

# Text vectorization for 'Overview'
tfidf_overview = TfidfVectorizer(stop_words='english')
tfidf_overview_matrix = tfidf_overview.fit_transform(movies['Overview'])

# Vectorization for 'Genre' with increased importance
tfidf_genre = TfidfVectorizer(stop_words='english')
tfidf_genre_matrix = tfidf_genre.fit_transform(movies['Genre'])
genre_weight = 50  # Increase this to put more emphasis on Genre
weighted_genre_matrix = tfidf_genre_matrix * genre_weight

# Vectorization for other textual data
tfidf_other = TfidfVectorizer(stop_words='english')
tfidf_other_matrix = tfidf_other.fit_transform(movies['Director'] + ' ' + \
                                               movies['Star1'] + ' ' + movies['Star2'] + ' ' + \
                                               movies['Star3'] + ' ' + movies['Star4'])

# Normalize numerical features after ensuring no NaNs are left
scaler = MinMaxScaler()
movies[['IMDB_Rating', 'Meta_score', 'No_of_Votes']] = scaler.fit_transform(
    movies[['IMDB_Rating', 'Meta_score', 'No_of_Votes']].fillna(0)  # Fill NaNs with 0 or handle appropriately
)

# Combine all features into a single similarity matrix
combined_features = np.hstack([weighted_genre_matrix.toarray(), tfidf_other_matrix.toarray(), tfidf_overview_matrix.toarray(),
                               movies[['IMDB_Rating', 'Meta_score', 'No_of_Votes']].values])

# Check for NaNs before computing similarity
if np.isnan(combined_features).any():
    print("NaN values detected in combined features.")
else:
    similarity_matrix = cosine_similarity(combined_features)

# Define the recommendation function
def get_recommendations(title, movies, similarity_matrix):
    # Get the index of the movie that matches the title
    idx = movies.index[movies['Series_Title'] == title].tolist()[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]  # Skip the first entry because it's the movie itself

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies['Series_Title'].iloc[movie_indices]

# Use the recommendation function to find movies similar to 'Monsters, Inc.'
recommended_movies = get_recommendations('Finding Nemo', movies, similarity_matrix)
print(recommended_movies)
