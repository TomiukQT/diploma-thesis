from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import pickle


from googletrans import Translator
import sys
sys.path.append("..")

from helpers.message import Message


class MessageTranslator:

    def __init__(self) -> None:
        self.translator = Translator()

    def translate_messages(self, messages: []) -> []:
        translated_messages = messages.copy()
        langs = [lang.lang for lang in self.translator.detect([msg.text for msg in messages])]
        for i, lang in enumerate(langs):
            if isinstance(lang, list):
                lang = lang[0]
            if lang != 'en':
                m = messages[i]
                try:
                    translated = self.translator.translate(m.text, src=str(lang), dest='en')
                except Exception:
                    translated = self.translator.translate(m.text, dest='en')
                translated_messages[i] = Message(translated.text, m.user, m.timestamp, m.reactions)
        return translated_messages


class DataTransformer:

    def __init__(self, path, vectorizer=None):
        self.stemmer = PorterStemmer()
        if path is not None:
            self.vectorizer = pickle.load(open(f'{path}', 'rb'))
        elif vectorizer is not None:
            self.vectorizer = vectorizer


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

    def __init__(self, path=None, vectorizer=None):
        DataTransformer.__init__(self, path, vectorizer)

    def vectorizer_fit(self, data: pd.Series):
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        self.vectorizer.fit(data)

    def transform(self, data: pd.Series, language: str = 'english'):
        tfidf_matrix = self.vectorizer.transform(data)
        return pd.DataFrame(tfidf_matrix.todense())


class BagOfWordsTransformer(DataTransformer):

    def __init__(self, path=None, vectorizer=None):
        DataTransformer.__init__(self, path, vectorizer)

    def vectorizer_fit(self, data: pd.Series):
        if self.vectorizer is None:
            self.vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=2000, stop_words='english')
        self.vectorizer.fit(data)

    def transform(self, data: pd.Series, language: str = 'english'):
        bow_matrix = self.vectorizer.transform(data)
        return pd.DataFrame(bow_matrix.todense())

