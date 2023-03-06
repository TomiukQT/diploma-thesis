from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import pickle

class DataTransformer:

    def __init__(self, path):
        self.stemmer = PorterStemmer()
        if path is not None:
            self.vectorizer = pickle.load(open(path, 'rb'))

    def stemming(self, data: pd.Series):
        tokenized = data.apply(lambda x: x.split())
        tokenized = tokenized.apply(lambda x: [self.stemmer.stem(i) for i in x])
        for i in range(len(tokenized)):
            tokenized[i] = ' '.join(tokenized[i])

        return tokenized

    def vectorizer_fit(self, data: pd.Series):
        pass

    def transform(self, data: pd.Series):
        pass


class TfidfDataTransformer(DataTransformer):

    def __init__(self, path=None):
        DataTransformer.__init__(self, path)

    def vectorizer_fit(self, data: pd.Series):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(data)

    def transform(self, data: pd.Series, language: str = 'english'):
        tfidf_matrix = self.vectorizer.transform(data)
        return pd.DataFrame(tfidf_matrix.todense())


class BagOfWordsTransformer(DataTransformer):

    def __init__(self, path=None, lang='english'):
        DataTransformer.__init__(self, path)
        self.lang = lang

    def vectorizer_fit(self, data: pd.Series):
        self.vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words=self.lang)
        self.vectorizer.fit(data)

    def transform(self, data: pd.Series, language: str = 'english'):
        bow_matrix = self.vectorizer.transform(data)
        return pd.DataFrame(bow_matrix.todense())

