"""Class docstring goes here.

"""
import math

from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np


class Utilities:
    @staticmethod
    def find_next_forecastdates(series, n_periods=1):
        """Find Multiple forecast dates for a given series and preds

        Args:
            series(pandas data frame) -- time series to perform forecast on
            n_periods
        """
        # Find the type of series object
        if (type(series).__name__ == "Series"):
            tsFreq = pd.infer_freq(series.index)
            last_date = series.index[-1]
            dates = pd.date_range(last_date, periods=n_periods, freq=tsFreq[0])
        return dates

    @staticmethod
    def addPeriodsToSeries(series, values):
        """Adding periods to a series.

        Args:
            series -- pandas series object
            values -- forecasted values to append to series
        """
        series = series.dropna(how='all')
        forecast_dates = Utilities.find_next_forecastdates(series, len(values))
        newseries = pd.Series(values, index=forecast_dates)
        result_series = series.append(newseries)
        return result_series

    @staticmethod
    def initial_trend(series, slen):
        s = 0.0
        for i in range(slen):
            s += float(series[i + slen] - series[i]) / slen
        return s / slen

    @staticmethod
    def initial_seasonal_components(series, slen):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(series) / slen)
        # compute season averages
        for j in range(n_seasons):
            season_averages.append(
                sum(series[slen * j:slen * j + slen]) / float(slen))
        # compute initial values
        for i in range(slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += series[slen * j +
                                               i] - season_averages[j]
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals
