import argparse

from content_based import recommend as recommend_content_based
from collaborative import recommend as recommend_collaborative
from combined import recommend as recommend_combined

parser = argparse.ArgumentParser(description='Movie recommendation engine.')

parser.add_argument('movie_title', type=str, help='Title of the movie for which you want to get the recommendations')

args = parser.parse_args()

print(f'Recommendations for the movie {args.movie_title}')

print('Content based recommendations:')
recommend_content_based(args.movie_title)

print('Collaborative recommendations:')
recommend_collaborative(args.movie_title)

print('Combined recommendations:')
recommend_combined(args.movie_title)
