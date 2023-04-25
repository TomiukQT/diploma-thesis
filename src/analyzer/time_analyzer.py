"""
This module will provide classes or/and functions to provide trend analysis and predictions.
It only works with cleaned data, max date_indexing will be applied??.
"""

import pandas as pd
import statsmodels.api as sm
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime


class TimeSeriesAnalyzer:

    @staticmethod
    def extract_trend(data, model='additive', start_date=None):
        """
        Extracts trend from a time series.

        :param start_date: The start date of the trend to view. Should be in the format 'YYYY-MM-DD' or a pandas Timestamp object.
        :param data: A time series with a datetime index.
        :param model: The decomposition model to use. 'additive' or 'multiplicative'.
        :return: Trend of time series
        """

        data = data.resample('1d')['value'].agg('mean').fillna(0).asfreq('1D')
        # Create a decomposition object with the specified model
        try:
            decomposition = sm.tsa.seasonal_decompose(data, model=model)
            # Extract the trend component from the decomposition object
            trend = decomposition.trend
        except Exception:
            print('Error while extracting trend')
            return None

        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            trend = trend.loc[start_date:]

        return trend.dropna()

    @staticmethod
    def get_predictions(data):
        try:
            data = data.resample('1d')['value'].agg('mean').fillna(method='ffill').asfreq('1D')
            stepwise_fit = pm.auto_arima(data, start_p=0, start_q=0,
                                         max_p=4, max_q=4, seasonal=False, d=0,
                                         information_criterion='aic',
                                         stepwise=True)
            model = ARIMA(data, order=stepwise_fit.order, trend='c')
            model = model.fit()
            predictions = model.get_prediction(end=data[-1:].index[0] + pd.Timedelta(days=10), dynamic=False)
            return predictions, data
        except Exception:
            print('Error while getting predictions')
            return None, None

    @staticmethod
    def parse_date(date_string):
        """

        :param date_string: string to be parsed. You can use
        'last_week',
        'last_month',
        'last_year',
        'this_week',  !!Does not work. Same as 'last_week'!!
        'this_month',
         'this_year'
        :return: pd.Timestamp(0) if something is wrong else correct pd.Timestamp converted from human-usable date specification.
        """
        ts = pd.Timestamp.today()
        try:
            if date_string == 'last_week':
                ts = pd.Timestamp.today() - pd.Timedelta(days=7)
            elif date_string == 'last_month':
                ts = pd.Timestamp.today() - pd.Timedelta(days=31)
            elif date_string == 'last_year':
                ts = pd.Timestamp.today() - pd.Timedelta(days=365)
            elif date_string == 'this_week':
                ts = pd.Timestamp.today() - pd.Timedelta(days=7)
            elif date_string == 'this_month':
                ts = pd.Timestamp.today() - pd.offsets.MonthBegin(1)
            elif date_string == 'this_year':
                ts = pd.Timestamp.today() - pd.offsets.YearBegin(1)
            return ts.floor('D')
        except Exception:
            return pd.Timestamp(0)

    @staticmethod
    def index_dates(data, dates):
        date_time = pd.to_datetime(dates)
        df = pd.DataFrame()
        df['value'] = data
        df = df.set_index(date_time)
        return df
