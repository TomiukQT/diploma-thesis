import pickle

import pandas as pd

from .data_cleaner import *
from .data_transformation import TfidfDataTransformer
import matplotlib.pyplot as plt


class Analyzer:

    def __init__(self, folder, model_name='model.sav', vectorizer_name='vectorizer.sav'):
        self.last_prediction = None
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
        #print(data)
        data = self.data_transformer.transform(data)
        #data = data[-data_len:]
        #print(data)
        predictions = self.model.predict_proba(data)
        self.last_prediction = [x[0] for x in predictions]
        return self.last_prediction

    @staticmethod
    def _emoji_from_score(score):
        return '😀' if score > 0.85 else '😐' if score > 0.75 else '☹'

    #TODO: Needed?????
    def index_dates(self, data, dates):
        date_time = pd.to_datetime(dates)
        df = pd.DataFrame()
        df['value'] = data
        df = df.set_index(date_time)
        return df

    def get_plot(self, plot_path=None, x_label='Date', y_label='Sentiment value', data2=pd.Series([-1, 0, 1])
                     , data3=pd.Series([-1, 0, 1])):
        if self.last_prediction is None:
            print('No predicitions done')
            return

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2), width_ratios=[3, 1, 1], sharey=True)
        fig.suptitle('Sentiment analysis', fontsize=14, y=1.1)

        data = pd.Series(self.last_prediction)
        # Plot sentiment values
        data.plot(ax=ax1)
        plt.gcf().autofmt_xdate()
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.set_title("Historical data")

        # Plot trend TODO
        data2.plot(ax=ax2)

        ax2.set_xlabel("Date")
        ax2.set_ylabel("Trend")
        ax2.set_title("Current trend")

        # Plot prediction?
        data3.plot(ax=ax3)

        ax3.set_xlabel("Date")
        ax3.set_ylabel("Prediction")
        ax3.set_title("Prediction")
        #ax3.text(4, 0, ''.join(data["value"].apply(lambda x: self.emoji_from_score(x))), fontsize=20)

        if plot_path is not None:
            plt.savefig(plot_path)
        return plot_path