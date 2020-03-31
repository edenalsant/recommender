from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import pandas as pd

def generate_tfidf_matrix(tfidf, dataset, col):
    return tfidf.fit_transform(dataset['overview'])

def generate_movies_indices(dataset, col):
    return pd.Series(dataset.index, index=dataset[col]).drop_duplicates()

def get_similar_movies_indeces(cosine_sim, index, amount):
    similar_score = list(enumerate(cosine_sim[index]))
    similar_score = sorted(similar_score, key=lambda x: x[1], reverse=True)
    last_movie_index = amount + 1
    return similar_score[1:last_movie_index]

def clear_dataset(dataset, col):
    return dataset[col].fillna('')

def get_similar_movies(dataset, movie_name, number_related_movies):
    tfidf = TfidfVectorizer(stop_words='english')
    
    dataset['overview'] = clear_dataset(dataset, 'overview')

    tfidf_matrix = generate_tfidf_matrix(tfidf, dataset, 'overview')

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = generate_movies_indices(dataset, 'title')

    movie_index = indices[movie_name]
    
    similar_score = get_similar_movies_indeces(cosine_sim, movie_index, number_related_movies)

    movie_indices = [i[0] for i in similar_score]
    return dataset['title'].iloc[movie_indices]