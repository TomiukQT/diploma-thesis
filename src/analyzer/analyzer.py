import pickle

import pandas as pd

from .data_cleaner import *
from .data_transformation import TfidfDataTransformer


class Analyzer:

    def __init__(self, folder, model_name='model.sav', vectorizer_name='vectorizer.sav'):
        self.model_path = f'{folder}/{model_name}'
        self.vectorizer_path = f'{folder}/{vectorizer_name}'
        self.model = None
        self.data_transformer = None

    def _load(self):
        if self.model is None:
            self.model = pickle.load(open(self.model_path, 'rb'))
        if self.data_transformer is None:
            self.data_transformer = TfidfDataTransformer(self.vectorizer_path)

    def analyze_sentence(self, text) -> (float, float):
        if self.model is None or self.data_transformer is None:
            self._load()
        predictions = self.model.predict_proba(text)

    def get_sentiment_analysis(self, texts: []) -> []:
        if self.model is None or self.data_transformer is None:
            self._load()
        data = pd.Series([clean_mentions(t) for t in texts])
        data = self.data_transformer.stemming(data)
        #data_len = len(data)
        #tr_data = pd.read_csv('models/model_data.csv')
        #tr_data.dropna(inplace=True)
        #tr_data = pd.concat([tr_data['clean_tweet'], data])
        data = self.data_transformer.transform(data)
        #data = data[-data_len:]
        #print(data)
        predictions = self.model.predict_proba(data)
        #print(predictions)
        return [x[0] for x in predictions]
