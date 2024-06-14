import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_imdb = pd.read_csv('imdb.csv')

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf_vectorizer.fit_transform(df_imdb['Genre'])
overview_matrix = tfidf_vectorizer.fit_transform(df_imdb['Overview'])
df_imdb['Actors'] = f"{df_imdb['Star1']}, {df_imdb['Star2']}, {df_imdb['Star3']}, {df_imdb['Star4']}"
actor_matrix = tfidf_vectorizer.fit_transform(df_imdb['Actors'])

# Combine all features into a single similarity matrix
combined_features = np.hstack([
    genre_matrix.toarray(),
    overview_matrix.toarray(),
    actor_matrix.toarray(),
])

# Check for NaNs before computing similarity
if np.isnan(combined_features).any():
    print("NaN values detected in combined features.")
    exit()
else:
    similarity_matrix = cosine_similarity(combined_features)

# Creating a Series with movie titles as the index
movie_indices = pd.Series(df_imdb.index, index=df_imdb['Series_Title']).drop_duplicates()

# Generate recommendations based on cosine similarity
def get_recommendations(movie_title):
    idx = movie_indices[movie_title]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = [score for score in similarity_scores if score[0] != idx]
    similarity_scores = similarity_scores[:10]
    movie_indices_list = [i[0] for i in similarity_scores]
    return df_imdb['Series_Title'].iloc[movie_indices_list][:10]

def get_recommend_list(movie_title):
    # Check if the movie exists in the dataset
    if movie_title in movie_indices:
        return get_recommendations(movie_title)
    else:
        return []

def recommend(movie_title):
    # Check if the movie exists in the dataset
    if movie_title in movie_indices:
        recommendations = get_recommendations(movie_title)
        for i, recommendation in enumerate(recommendations, 1):
            print(f"{i}. {recommendation}")
    else:
        print("Movie not found")

if __name__ == "__main__":
    movie_title = 'The Shawshank Redemption'
    recommend(movie_title)
