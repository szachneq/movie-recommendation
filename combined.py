import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from content_based import get_recommend_list as get_content_based
from collaborative import get_recommend_list as get_collaborative

def recommend(movie_title):
    results = [ *get_content_based(movie_title)[:5], *get_collaborative(movie_title)[:5] ]
    for i, recommendation in enumerate(results, 1):
        print(f"{i}. {recommendation}")

if __name__ == "__main__":
    movie_title = 'The Shawshank Redemption'
    recommend(movie_title)
