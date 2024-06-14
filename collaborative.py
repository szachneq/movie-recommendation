import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df_imdb = pd.read_csv('imdb.csv')
df_rating = pd.read_csv('ratings.csv')

# Identify movies that exist in both datasets
common_movies = set(df_imdb['Series_Title']).intersection(set(df_rating['Series_Title']))
# Filter both datasets to include only movies found in both
df_imdb = df_imdb[df_imdb['Series_Title'].isin(common_movies)]
df_rating = df_rating[df_rating['Series_Title'].isin(common_movies)]

# Merge movie and rating data
data = pd.merge(df_rating, df_imdb, on='Series_Title', how='inner')
# Create a user-item matrix
user_item_matrix = data.pivot_table(index='user_id', columns='Series_Title', values='rating')
user_item_matrix = user_item_matrix.fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Generate recommendations based on cosine similarity
def get_recommendations(movie_title):
    # Get the index of users who like the movie
    movie_users = user_item_matrix[user_item_matrix[movie_title] > 0].index
    # Get the similarity scores for these users
    similarity_scores = user_similarity_df.loc[movie_users].mean(axis=0)
    # Sort the scores
    similarity_scores = similarity_scores.sort_values(ascending=False)
    # Get top similar users
    top_users = similarity_scores.index[1:11]  # exclude the target user itself
    # Get movie recommendations from other similar users
    recommended_movies = user_item_matrix.loc[top_users].mean(axis=0)
    recommended_movies = recommended_movies.sort_values(ascending=False)
    
    return recommended_movies.index[:10]

def recommend(movie_title):
    if movie_title in user_item_matrix.columns:
        recommendations = get_recommendations(movie_title)
        for i, recommendation in enumerate(recommendations, 1):
            print(f"{i}. {recommendation}")
    else:
        print("Movie not found")


if __name__ == "__main__":
    movie_title = 'The Shawshank Redemption'
    recommend(movie_title)
