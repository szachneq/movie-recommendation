# import pandas as pd

# # Load the data from CSV files
# df_movies = pd.read_csv('movies_metadata.csv', usecols=['id', 'original_title'])  # Replace 'path_to_first_csv.csv' with the actual path to your CSV
# df_ratings = pd.read_csv('ratings_small.csv', usecols=['userId', 'movieId', 'rating'])  # Replace 'path_to_second_csv.csv' with the actual path to your CSV

# # Rename the columns for clarity, if necessary
# df_movies.columns = ['id', 'original_title']  # Adjust column names if they are different
# df_ratings.columns = ['userId', 'id', 'rating']  # Adjust column names if they are different

# # Merge the dataframes on the movieId column
# df_merged = pd.merge(df_ratings, df_movies, on='id', how='left')

# # Save the merged data back to CSV
# df_merged.to_csv('merged_ratings.csv', index=False)

# print("Merging completed. The output is saved as 'merged_ratings.csv'.")

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
df_merged.to_csv('merged_ratings.csv', index=False)

print("Merging completed. The output is saved as 'merged_ratings.csv'.")

# ----------

# Calculate C, the mean of all ratings in the dataset
C = df_merged['rating'].mean()

# Calculate the number of ratings for each movie and the average rating
rating_stats = df_merged.groupby('Series_Title')['rating'].agg(['count', 'mean'])

# Calculate m as the minimum votes required to be considered, e.g., the 50th percentile
m = rating_stats['count'].quantile(0.50)

# Define the weighted rating calculation
def weighted_rating(x, m=m, C=C):
    v = x['count']
    R = x['mean']
    return (v/(v+m) * R) + (m/(m+v) * C)

# Apply the weighted rating formula
rating_stats['weighted_rating'] = rating_stats.apply(weighted_rating, axis=1)

# Sort the results by the weighted rating in descending order
weighted_ratings = rating_stats['weighted_rating'].sort_values(ascending=False)

# Print the sorted weighted ratings
# print(weighted_ratings)
weighted_ratings.to_csv('weighted_ratings.csv')

# average_ratings = df_merged.groupby('Series_Title')['rating'].mean()

# # Sort the results by movie title
# average_ratings = average_ratings.sort_values(ascending=False)

# # Save the data
# average_ratings.to_csv('average_ratings.csv')