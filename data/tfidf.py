from sklearn.feature_extraction.text import TfidfVectorizer

def remove_stop_words(lang):
     return TfidfVectorizer(stop_words=lang)

def replace_nan(dataset, col, string):
    dataset[col] = dataset[col].fillna(string)

def build_matrix(dataset, col):
    tfidf = remove_stop_words('english')
    replace_nan(dataset, col, '')
    return tfidf.fit_transform(dataset[col])

