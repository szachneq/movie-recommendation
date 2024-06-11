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

# Load user preferences
user_preferences_path = 'user_preferences.csv'
user_prefs = pd.read_csv(user_preferences_path)

def get_user_recommendations(user_id, user_prefs, movies, similarity_matrix):
    # Filter to get this user's preferences
    user_movies = user_prefs[user_prefs['user_id'] == user_id]

    # Get indices for movies watched by this user
    watched_movies = movies[movies['Series_Title'].isin(user_movies['Series_Title'])]
    watched_indices = watched_movies.index.tolist()

    # Initialize a similarity score array with zeros
    weighted_sim_scores = np.zeros(similarity_matrix.shape[0])

    # Aggregate weighted similarity scores based on user ratings
    for index, row in watched_movies.iterrows():
        movie_idx = index
        movie_rating = user_movies[user_movies['Series_Title'] == row['Series_Title']]['rating'].iloc[0]
        weighted_sim_scores += similarity_matrix[movie_idx] * movie_rating

    # Normalize by the number of ratings to prevent bias towards users with more ratings
    if not watched_indices:
        return "User has not rated any movies."
    weighted_sim_scores /= len(watched_indices)

    # Exclude already watched movies from recommendations
    weighted_sim_scores[watched_indices] = 0

    # Get indices of the top 10 movies
    recommended_indices = np.argsort(weighted_sim_scores)[::-1][:10]

    # Get recommended movie titles
    recommended_movies = movies['Series_Title'].iloc[recommended_indices]

    return recommended_movies



user_id = 1
recommended_movies = get_user_recommendations(user_id, user_prefs, movies, similarity_matrix)
print(recommended_movies)