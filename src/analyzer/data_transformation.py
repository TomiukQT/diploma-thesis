from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

class DataTransformer:

    def __init__(self, path):
        self.stemmer = PorterStemmer()
        self.vectorizer = pickle.load(open(path, 'rb'))

    def stemming(self, data: pd.Series):
        tokenized = data.apply(lambda x: x.split())
        tokenized = tokenized.apply(lambda x: [self.stemmer.stem(i) for i in x])
        for i in range(len(tokenized)):
            tokenized[i] = ' '.join(tokenized[i])

        return tokenized

    def transform(self, data: pd.Series):
        pass


class TfidfDataTransformer(DataTransformer):

    def __init__(self, path):
        DataTransformer.__init__(self, path)

    def transform(self, data: pd.Series, language: str = 'english'):
        tfidf_matrix = self.vectorizer.transform(data)
        return pd.DataFrame(tfidf_matrix.todense())

