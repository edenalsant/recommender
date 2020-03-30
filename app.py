from data import loader
from data import ranking
from data import tfidf

def run():
    movies_dataset = loader.load_dataset('./resources/movies_metadata.csv')
    
    general_mean = ranking.get_mean(movies_dataset, "vote_average")
    minimum_votes_needed = ranking.get_minmum_votes_needed(movies_dataset, 0.9)

    eligible_movies = ranking.get_eligible_movies(movies_dataset, minimum_votes_needed)

    weighted_rating = ranking.get_weighted_rating(eligible_movies, minimum_votes_needed, general_mean)
    
    eligible_movies['score'] = weighted_rating
    eligible_movies = eligible_movies.sort_values('score', ascending=False)

    tfidf_matrix = tfidf.build_matrix(eligible_movies, 'overview')
    print(tfidf_matrix.shape)


