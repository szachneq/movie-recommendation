import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load movie data
data_path = 'imdb.csv'
movies = pd.read_csv(data_path)

# Clean and preprocess movie data
movies = movies.drop_duplicates(subset='Series_Title', keep='first')
movies = movies.dropna(subset=['Overview', 'Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4'])
movies = movies.drop(columns=['Poster_Link'])

# Feature Engineering for text data
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['Overview'])
movies['all_text'] = movies['Genre'] + ' ' + movies['Director'] + ' ' + \
                     movies['Star1'] + ' ' + movies['Star2'] + ' ' + \
                     movies['Star3'] + ' ' + movies['Star4']
text_matrix = tfidf.fit_transform(movies['all_text'])

# Normalize numerical features
scaler = MinMaxScaler()
movies[['IMDB_Rating', 'Meta_score', 'No_of_Votes']] = scaler.fit_transform(
    movies[['IMDB_Rating', 'Meta_score', 'No_of_Votes']].fillna(0)
)

# Combine all features into a single similarity matrix for content-based filtering
combined_features = np.hstack([text_matrix.toarray(), tfidf_matrix.toarray(),
                               movies[['IMDB_Rating', 'Meta_score', 'No_of_Votes']].values])

# Compute similarity matrix for content-based features
similarity_matrix = cosine_similarity(combined_features)

# Load ratings data
ratings_path = 'ratings.csv'
ratings = pd.read_csv(ratings_path)

# Pivot ratings into matrix format
ratings_matrix = ratings.pivot_table(index='user_id', columns='Series_Title', values='rating').fillna(0)

# Compute similarity matrix for collaborative filtering
user_similarity = cosine_similarity(ratings_matrix)
user_similarity = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)

def predict_ratings(user_id):
    # Predict ratings for each movie based on user similarity
    similar_users = user_similarity[user_id]
    user_ratings = ratings_matrix.loc[user_id]
    pred_ratings = ratings_matrix.T.dot(similar_users) / similar_users.sum()
    return pred_ratings

def get_combined_recommendations(title, user_id, movies, similarity_matrix, ratings_matrix):
    # Content-based filtering
    idx = movies.index[movies['Series_Title'] == title].tolist()[0]
    content_scores = list(enumerate(similarity_matrix[idx]))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)[1:11]

    # Collaborative filtering
    predicted_ratings = predict_ratings(user_id)
    collab_scores = list(enumerate(predicted_ratings))
    collab_scores = sorted(collab_scores, key=lambda x: x[1], reverse=True)[1:11]

    # Combine scores from both models (weighted average or similar method can be applied)
    combined_scores = [(content[0], 0.5 * content[1] + 0.5 * collab[1]) for content, collab in zip(content_scores, collab_scores)]
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)

    # # Exclude movies already watched by the user
    watched_movies = set(ratings_matrix.columns[ratings_matrix.loc[user_id] > 0])  # Movies with a rating

    # Get the top movies based on combined score
    movie_indices = [i[0] for i in combined_scores]
    recommendations = movies['Series_Title'].iloc[movie_indices]

    # remove revommendations if the movie was watched before
    filtered_recommendations = []
    for recommendation in recommendations:
        if recommendation in watched_movies:
            continue
        else:
            filtered_recommendations.append(recommendation)

    return filtered_recommendations

# Example of usage
user_id = 1  # Example user ID
recommended_movies = get_combined_recommendations('The Shawshank Redemption', user_id, movies, similarity_matrix, ratings_matrix)
print(recommended_movies)
