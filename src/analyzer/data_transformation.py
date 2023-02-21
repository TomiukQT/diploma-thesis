from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class DataTransformer:

    def __init__(self):
        self.stemmer = PorterStemmer()

    def stemming(self, data: pd.Series):
        tokenized = data.apply(lambda x: x.split())
        tokenized = tokenized.apply(lambda x: [self.stemmer.stem(i) for i in x])
        for i in range(len(tokenized)):
            tokenized[i] = ' '.join(tokenized[i])

        return tokenized

    def transform(self, data: pd.Series):
        pass


class TfidfDataTransformer(DataTransformer):

    def __init__(self):
        DataTransformer.__init__()

    def transform(self, data: pd.Series, language: str = 'english'):
        tfidf = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data)
        return pd.DataFrame(tfidf_matrix.todense())

