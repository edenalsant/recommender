from data import loader
from data import ranking
from data import tfidf

def get_movies_ranking(dataset):
    return ranking.rank_movies(dataset)

def get_recomendations(dataset, movie_name, number_related_movies):
    return tfidf.get_similar_movies(dataset, movie_name, number_related_movies) 

def run():
    movies_dataset = loader.load_dataset('./resources/movies_metadata.csv')
    movies_dataset = get_movies_ranking(movies_dataset)
    print('these are the top 20 movies: ')
    print(movies_dataset['title'].head(20))

    movie_name = input('movie title: ')
    number_of_movies = int(input('how many movies? '))
    print(get_recomendations(movies_dataset, movie_name, number_of_movies))
