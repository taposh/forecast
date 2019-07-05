import pandas as pd
from kpforecast.utils import Utilities


class MovingAvg(object):
    @staticmethod
    def moving_average_forecast(series, window_size, n_periods=1):
        """Moving Average Forecast.

        Args:
            series - pandas series object
            window - sliding window size
        Returns:
        """
        result_series = Utilities.average_forecast_one_period(
            series[-window_size:])
        return result_series

    @staticmethod
    def moving_average_forecast_n_periods(series,
                                          window_size,
                                          n,
                                          start_date):
        """Moving Average Forecast

        Args:
            series -- pandas series object
            window -- sliding window size
            n -- number of periods
        """
        if window_size > len(series):
            raise Exception("Window size should be less than series")
        X = series.to_numpy()
        # Construct the historic and test sets
        history = [X[i] for i in range(window_size)]
        # number of items to predict
        predictions = []
        # walk forward over time steps in test
        for _ in range(n):
            length = len(history)
            yhat = np.mean(
                [history[i] for i in range(length - window_size, length)])
            predictions.append(yhat)
        tsFreq = pd.infer_freq(series.index)
        dates = pd.date_range(start_date, periods=n, freq=tsFreq[0])
        pred_series = pd.Series(predictions, dates)
        result_series = pd.concat([series, pred_series])
        return result_series

    @staticmethod
    def moving_average_forecast_n_periods_old(series,
                                              window_size,
                                              n_periods,
                                              start_date,
                                              verbose=True):
        """Moving Average Forecast for number of periods.

        Args:
            series -- pandas series object
            window_size -- sliding window size
            n_periods -- number of periods
            start_date -- start date for the series
        """
        # Get forecast for 1 value
        result_series = Utilities.average_forecast_one_period(
            series[-window_size:])
        # input series
        input_series = result_series
        # remaining n-1 periods
        for ii in range(n_periods):
            result_series = Utilities.average_forecast_one_period(
                input_series[-window_size:])
            input_series = pd.concat(
                [input_series, result_series[len(result_series) - 1:]])
        return input_series
