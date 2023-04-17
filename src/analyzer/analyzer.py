import pickle
import statistics
import json

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
        self.last_date_results = None
        self.last_reactions = None
        self.last_results = None
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

    def _is_proba_model(self):
        method = getattr(self.model, 'predict_proba', None)
        return callable(method)

    def analyze_sentence(self, text) -> (float, float):
        if self.model is None or self.data_transformer is None:
            self._load()
        data = pd.Series([clean_mentions(text)])
        data = self.data_transformer.stemming(data)
        data = self.data_transformer.transform(data)

        if self._is_proba_model():
            predictions = self.model.predict_proba(data)
            return predictions[0][0]
        else:
            predictions = self.model.predict(data)
            return predictions[0]

    def get_sentiment_analysis(self, messages: []) -> []:
        """
        Analyze sentiment from input.
        :param messages: Input texts/messages
        :return: Sentiment analysis output
        """
        #
        if self.model is None or self.data_transformer is None:
            self._load()
        texts = [m.text for m in messages]
        reactions = [m.reactions for m in messages]

        data = pd.Series([clean_mentions(t) for t in texts])
        data = self.data_transformer.stemming(data)
        data = self.data_transformer.transform(data)
        if self._is_proba_model():
            predictions = self.model.predict_proba(data)
        else:
            predictions = self.model.predict(data)
        self.last_prediction = predictions
        self.last_reactions = self.evaluate_reactions(reactions)
        self.last_results = self.metrics()
        self.last_date_results = self.index_dates(self.last_results, [m.date.date() for m in messages])

        return self.last_results

    def evaluate_reactions(self, reactions_data: []):
        results = []
        reactions_cfg = json.load(open('reactions_cfg.json'))
        for reactions in reactions_data:
            if reactions is None:
                results.append(0)
            else:
                score = 0
                total_weight = sum([r.count for r in reactions])
                for reaction in reactions:
                    if reaction.name in reactions_cfg:
                        score += reactions_cfg[reaction.name] * reaction.count / total_weight
                    else:
                        score += self.analyze_sentence(reaction.name) * reaction.count / total_weight
                results.append(score)
        return results

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

    def get_plot(self, plot_path=None, trend_data=pd.Series([-1, 0, 1])
                 , predictions_data=None):
        """

        :param plot_path: Path, where final plot will be saved
        :param x_label: Label of X axis of main plot. This parameter will be probably removed.
        :param y_label: Label of Y axis of main plot. This parameter will be probably removed.
        :param trend_data: Data for trend plot.
        :param predictions_data: Data for prediction plot
        :return: Path where plot is saved. None if no plot was saved.
        """
        if self.last_results is None:
            print('No predictions done')
            return
        data = self.last_date_results.resample('D').mean().fillna(0)
        #data = pd.Series(self.last_results)
        figsize = (5, 3)
        plt.style.use('seaborn-v0_8-whitegrid')
        # SENTIMENT
        fig, ax = plt.subplots(figsize=figsize)
        plt.ylim(-1, 1)
        data.plot(ax=ax, label='Sentiment', marker='.')

        # Labels
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment')
        ax.set_title("Historical data")
        plot_path1 = self.__save__plot(plt, plot_path, 1)
        # TREND
        fig, ax = plt.subplots(figsize=figsize)
        if trend_data is None or len(trend_data) == 0:
            ax.text(0.5, 0.5, "No data", fontsize=20)
        else:
            pd.Series(trend_data).plot(ax=ax)
            ax.set_xlabel("Date")
            ax.set_ylabel("Trend")
            ax.set_title("Current trend")
        plot_path2 = self.__save__plot(plt, plot_path, 2)


        # PREDICTIONS
        fig, ax = plt.subplots(figsize=figsize)
        if predictions_data is None or len(predictions_data) == 0:
            ax.text(0.5, 0.5, "No data", fontsize=20)
        else:
            plt.ylim(-1, 1)
            if predictions_data is None or len(predictions_data) == 0:
                pd.Series([-1, 0, 1]).plot(ax=ax)

            data_len = len(data)

            predictions, original = predictions_data
            predictions.predicted_mean[data_len-1:].plot(label='predictions', marker='.', ax=ax)
            ci = predictions.conf_int()
            ci[data_len-1:].plot(color='grey', ax=ax)
            original[-2:].plot(label='orig_data', marker='.', ax=ax)
            plt.legend()

            ax.set_xlabel("Date")
            ax.set_ylabel("Prediction")
            ax.set_title("Prediction")
            ax.set_ylim(-1, 1)

        plot_path3 = self.__save__plot(plt, plot_path, 3)
        return plot_path1, plot_path2, plot_path3

    @staticmethod
    def __save__plot(_plt_, plot_path, i):
        if plot_path is not None:
            path = plot_path + str(i) + '.png'
            _plt_.savefig(path)
        return path

    def metrics(self, reaction_weight=0.2, data=None):
        """
        Calculate metrics for prediction.
        :return: Metric
        """
        if data is None:
            data = self.last_prediction
        if data is None:
            print('No predictions done')
            return
        prediction_weight = 1 - reaction_weight
        if self._is_proba_model():
            normalized_prediction = list(map(lambda x: (x - 0.5) * 2, self.last_prediction[:, 0]))
        else:
            normalized_prediction = list(map(lambda x: ((1-x) - 0.5) * 2, self.last_prediction))
        return list(map(lambda x, y: x * prediction_weight + y * reaction_weight, normalized_prediction, self.last_reactions))
