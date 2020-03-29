import pandas as pd

def load_dataset(path):
    return pd.read_csv(path, low_memory=False)

def get_mean(dataset, col):
    return dataset[col].mean()

def get_minmum_votes_needed(dataset, amount):
    return dataset['vote_count'].quantile(amount)

def get_eligible_movies(dataset, minimum_voted_needed):
    return dataset.copy().loc[dataset['vote_count'] >= minimum_voted_needed]

def get_weighted_rating(eligible_movies, minimum_voted_needed, general_mean):
    v = eligible_movies['vote_count']
    R = eligible_movies['vote_average']
    return ((v/(v + minimum_voted_needed) * R) + (minimum_voted_needed/(minimum_voted_needed+v) * general_mean))

def add_colum_to_dataset(dataset, column, weighted_rating):
    dataset[column] = dataset.apply(weighted_rating, axis=1)
    return dataset

def main():
    movies_dataset = load_dataset('the-movies-dataset/movies_metadata.csv')

    general_mean = get_mean(movies_dataset, "vote_average")
    minimum_votes_needed = get_minmum_votes_needed(movies_dataset, 0.9)

    eligible_movies = get_eligible_movies(movies_dataset, minimum_votes_needed)

    weighted_rating = get_weighted_rating(eligible_movies, minimum_votes_needed, general_mean)
    
    eligible_movies['score'] = weighted_rating
    eligible_movies = eligible_movies.sort_values('score', ascending=False)
if __name__ == "__main__":
    main()