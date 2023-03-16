"""
This module will provide classes or/and functions to provide trend analysis and predictions.
It only works with cleaned data, max date_indexing will be applied??.
"""

import pandas as pd
import statsmodels.api as sm


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
        data = data.asfreq('D')
        # Create a decomposition object with the specified model
        decomposition = sm.tsa.seasonal_decompose(data, model=model)

        # Extract the trend component from the decomposition object
        trend = decomposition.trend

        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            trend = trend.loc[start_date:]

        return trend

    @staticmethod
    def _parse_date(date_string):
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
        ts = pd.Timestamp(date_string)
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
