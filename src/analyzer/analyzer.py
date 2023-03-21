import pickle
import pandas as pd
import matplotlib.pyplot as plt

from .data_cleaner import *
from .data_transformation import TfidfDataTransformer


class Analyzer:
    """
    Analyzer, which uses already pretrained model to analyze sentiment from given texts (messages).
    """

    def __init__(self, folder, model_name='model.sav', vectorizer_name='vectorizer.sav'):
        """
        :param folder: Folder path, where serialized models/vectorizers located
        :param model_name: Name of serialized model file
        :param vectorizer_name: Name of serialized vectorizer file
        """
        self.last_prediction = None
        self.model_path = f'{folder}/{model_name}'
        self.vectorizer_path = f'{folder}/{vectorizer_name}'
        self.model = None
        self.data_transformer = None

    def _load(self):
        """
        Deserialize and assign model, create new DataTransformer.
        :return:
        """
        if self.model is None:
            self.model = pickle.load(open(self.model_path, 'rb'))
        if self.data_transformer is None:
            self.data_transformer = TfidfDataTransformer(self.vectorizer_path)

    def analyze_sentence(self, text) -> (float, float):
        if self.model is None or self.data_transformer is None:
            self._load()
        predictions = self.model.predict_proba(text)

    def get_sentiment_analysis(self, texts: []) -> []:
        """
        Analyze sentiment from input.
        :param texts: Input texts/messages
        :return: Sentiment analysis output
        """
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
        """
        Score to emoji converted
        :param score: Input score
        :return: Emoji based on score
        """
        return '😀' if score > 0.85 else '😐' if score > 0.75 else '☹'

    #TODO: Needed?????
    @staticmethod
    def index_dates(data, dates):
        date_time = pd.to_datetime(dates)
        df = pd.DataFrame()
        df['value'] = data
        df = df.set_index(date_time)
        return df

    def get_plot(self, plot_path=None, x_label='Date', y_label='Sentiment value', trend_data=pd.Series([-1, 0, 1])
                 , predictions_data=None):
        """

        :param plot_path: Path, where final plot will be saved
        :param x_label: Label of X axis of main plot. This parameter will be probably removed.
        :param y_label: Label of Y axis of main plot. This parameter will be probably removed.
        :param trend_data: Data for trend plot.
        :param predictions_data: Data for prediction plot
        :return: Path where plot is saved. None if no plot was saved.
        """
        if self.last_prediction is None:
            print('No predicitions done')
            return

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2), width_ratios=[3, 1, 3 ], sharey=True)
        fig.suptitle('Sentiment analysis', fontsize=14, y=1.1)

        data = pd.Series(self.last_prediction)
        # Plot sentiment values
        data.plot(ax=ax1)
        plt.gcf().autofmt_xdate()
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.set_title("Historical data")

        # Plot trend TODO
        trend_data.plot(ax=ax2)
        plt.gcf().autofmt_xdate()
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Trend")
        ax2.set_title("Current trend")

        # Plot prediction?
        if predictions_data is None:
            pd.Series([-1, 0, 1]).plot(ax=ax3)
        predictions, original = predictions_data
        predictions.predicted_mean.plot(label='predictions', ax=ax3)
        ci = predictions.conf_int()
        ci.plot(color='grey', ax=ax3)
        original.plot(label='data', marker='.', ax=ax3)

        plt.ylim(-1, 1)

        ax3.set_xlabel("Date")
        ax3.set_ylabel("Prediction")
        ax3.set_title("Prediction")
        #ax3.text(4, 0, self._emoji_from_score(data[-1]), fontsize=20)

        if plot_path is not None:
            plt.savefig(plot_path)
        return plot_path
