import pandas as pd

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