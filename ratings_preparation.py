import pandas as pd

# Load the data from CSV files
df_movies = pd.read_csv('movies_metadata.csv', usecols=['id', 'original_title'])
df_ratings = pd.read_csv('ratings_small.csv', usecols=['userId', 'movieId', 'rating'])

# Convert 'id' in df_movies to integer, ensuring all are valid integers
df_movies['id'] = pd.to_numeric(df_movies['id'], errors='coerce')
df_movies.dropna(subset=['id'], inplace=True)  # Drop any rows where 'id' could not be converted
df_movies['id'] = df_movies['id'].astype(int)  # Convert to int

# Ensure 'movieId' in df_ratings is also an integer
df_ratings['movieId'] = df_ratings['movieId'].astype(int)

# Merge the dataframes on the movieId/id column
df_merged = pd.merge(df_ratings, df_movies, left_on='movieId', right_on='id', how='left')

# Drop the now redundant 'id' column from movies
df_merged.drop('id', axis=1, inplace=True)
df_merged.drop('movieId', axis=1, inplace=True)

# Optionally rename 'original_title' back to something indicating it's the movie ID now
df_merged.rename(columns={'userId': 'user_id'}, inplace=True)
df_merged.rename(columns={'original_title': 'Series_Title'}, inplace=True)

df_merged = df_merged.dropna(subset=['Series_Title'])

df_merged = df_merged[['user_id', 'Series_Title', 'rating']]

# Save the merged data back to CSV
df_merged.to_csv('ratings.csv', index=False)

print("Merging completed. The output is saved as 'ratings.csv'.")
