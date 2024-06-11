import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from surprise import Reader, Dataset, SVD

# Load datasets
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Replace '|' with space in genres for easier processing
movies_df['genres'] = movies_df['genres'].str.replace('|', ' ')

# Content-Based Filtering: Using movie genres
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(movies_df['genres'])
cosine_sim = cosine_similarity(X, X)

# Collaborative Filtering: Using SVD from Surprise
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
svd = SVD()
trainset = data.build_full_trainset()
svd.fit(trainset)

# Function to get recommendations based on a list of movie IDs
def hybrid_recommendation_by_movie_ids(movie_ids, num_recommendations=5):
    # Content-based recommendations
    top_similar = []
    for movie_id in movie_ids:
        if movie_id in movies_df['movieId'].values:
            movie_idx = movies_df.index[movies_df['movieId'] == movie_id].tolist()[0]
            sim_scores = list(enumerate(cosine_sim[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
            top_similar += sim_scores
    
    # Collate similar scores and sort them
    movie_indices = [i[0] for i in top_similar]
    unique_movie_indices = list(set(movie_indices))
    
    # Get top N movie indices based on similarity score
    weighted_scores = []
    for idx in unique_movie_indices:
        sim_scores = np.array([score for i, score in top_similar if i == idx])
        weighted_score = np.mean(sim_scores)
        # Generate a "mock" user rating for collaborative filtering
        # Assume the user would rate movies similar to their list highly
        mock_rating = 4.0  # This is an assumption
        weighted_score *= (1 + 0.1 * mock_rating)
        weighted_scores.append((idx, weighted_score))
    
    weighted_scores = sorted(weighted_scores, key=lambda x: x[1], reverse=True)
    
    # Get movie IDs from the top recommendations
    top_movie_indices = [i[0] for i in weighted_scores[:num_recommendations]]
    recommended_movie_ids = movies_df.iloc[top_movie_indices]['movieId'].tolist()
    
    # Return top recommended movie titles
    return movies_df[movies_df['movieId'].isin(recommended_movie_ids)]['title']

# Childrens movies
mock_movie_ids = [
    1, # Toy Story
    # 3114, # Toy Story 2
    # 78499, # Toy Story 3
    6377, # Finding Nemo
    # 157296, # Finding Dory
    4306, # Shrek
    2081, # The Little Mermaid
    45517, # Cars
]
recommended_movies = hybrid_recommendation_by_movie_ids(mock_movie_ids, 5)
print(recommended_movies)

# Horror movies
mock_movie_ids = [
    593, # Silence of the Lambs
    1214, # Alien
    1219, # Psycho
    1258, # The Shining
]
recommended_movies = hybrid_recommendation_by_movie_ids(mock_movie_ids, 5)
print(recommended_movies)

# Comedies
mock_movie_ids = [
    20, # Ace Ventura
    1080, # Monty Python's Life of Brian
    92259, # Intouchables
]
recommended_movies = hybrid_recommendation_by_movie_ids(mock_movie_ids, 5)
print(recommended_movies)

# nie rekomenduj obejrzanego filmu
# wyliczenie total score z ocen uzytkownikow

