import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
df_imdb = pd.read_csv('imdb.csv')
df_rating = pd.read_csv('ratings.csv')

# Process Content-based Features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf_vectorizer.fit_transform(df_imdb['Genre'])
overview_matrix = tfidf_vectorizer.fit_transform(df_imdb['Overview'])
df_imdb['Actors'] = f"{df_imdb['Star1']}, {df_imdb['Star2']}, {df_imdb['Star3']}, {df_imdb['Star4']}"
actor_matrix = tfidf_vectorizer.fit_transform(df_imdb['Actors'])
combined_features = np.hstack([
    genre_matrix.toarray(),
    overview_matrix.toarray(),
    actor_matrix.toarray(),
])

# Compute similarity matrix for content-based features
similarity_matrix_content = cosine_similarity(combined_features)
movie_indices = pd.Series(df_imdb.index, index=df_imdb['Series_Title']).drop_duplicates()

# Process Collaborative Filtering Features
common_movies = set(df_imdb['Series_Title']).intersection(set(df_rating['Series_Title']))
df_imdb = df_imdb[df_imdb['Series_Title'].isin(common_movies)]
df_rating = df_rating[df_rating['Series_Title'].isin(common_movies)]
data = pd.merge(df_rating, df_imdb, on='Series_Title', how='inner')
user_item_matrix = data.pivot_table(index='user_id', columns='Series_Title', values='rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def hybrid_recommend(movie_title):
    # Check if the movie exists for content-based recommendation
    if movie_title in movie_indices:
        idx = movie_indices[movie_title]
        content_scores = list(enumerate(similarity_matrix_content[idx]))
        content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)[1:11]  # Top 10, excluding self

        # Resetting indices to ensure alignment
        df_imdb.reset_index(drop=True, inplace=True)

        # Safely accessing movie titles with valid indices
        content_recommendations = [df_imdb['Series_Title'].iloc[i[0]] for i in content_scores if i[0] < len(df_imdb)]
    
    # Check if the movie exists for collaborative filtering recommendation
    if movie_title in user_item_matrix.columns:
        movie_users = user_item_matrix[user_item_matrix[movie_title] > 0].index
        collab_scores = user_similarity_df.loc[movie_users].mean(axis=0).sort_values(ascending=False)[1:11]
        collab_recommendations = user_item_matrix.loc[collab_scores.index].mean(axis=0).sort_values(ascending=False).index[:10]
    
    # Combine and rank recommendations if both lists are initialized
    if 'content_recommendations' in locals() and 'collab_recommendations' in locals():
        combined_recommendations = pd.Series(np.append(content_recommendations, collab_recommendations))
        final_recommendations = combined_recommendations.value_counts().index[:10]  # Top 10 unique recommendations
        return final_recommendations
    else:
        return "Unable to generate recommendations due to data issues."

def recommend(movie_title):
    if movie_title in movie_indices or movie_title in user_item_matrix.columns:
        recommendations = hybrid_recommend(movie_title)
        if isinstance(recommendations, pd.Index):
            for i, recommendation in enumerate(recommendations, 1):
                print(f"{i}. {recommendation}")
        else:
            print(recommendations)
    else:
        print("Movie not found")


if __name__ == "__main__":
    movie_title = 'The Shawshank Redemption'
    recommend(movie_title)
