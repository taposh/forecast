from kpforecast.utils import Utilities

import numpy as np
import copy
from scipy import stats.linregress as linregress


class Statistical():
    """
    This module represents a collection of common statistical
    forecasting methods
    """

    @staticmethod
    def naive_f(series, n_periods):
        """naive forecasting for n periods
        Args:
            series(pandas data frame) -- time series to perform forecast
            on.
            n_periods(int): number of periods
        """
        myseries = series.dropna(how='all')
        values = [myseries[-1]] * n_periods
        return Utilities.addPeriodsToSeries(myseries, values)

    @staticmethod
    def avg_f(series, n_periods=1):
        """Average forecasting for one periods
        Args:
            series(pandas data frame) -- time series to perform forecast
            on.
            n_periods(int): number of periods
        """
        myseries = series.dropna(how='all')
        expS1 = list(myseries.index)
        avg_values = myseries.to_numpy()[:]
        for _ in range(n_periods):
            avg_values.append(np.mean(avg_values))
        return pd.Series(avg_values, index=expS1)

    @staticmethod
    def weighted_avg_f(series, weights, n_periods=1):
        """Weighted Average Forecast.

        Args:
            series(pandas data frame) -- time series to perform forecast on
            weights(list): list of weights to use when computing weighted avg
        """
        result = 0.0
        weights.reverse()

        itr = len(weights) if len(weights) < len(series) else len(series)
        for n in range(itr):
            result += series[-n - 1] * weights[n]
        return Utilities.addPeriodsToSeries(series, result)

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
    def ses_f(series, alpha, n_periods=1):
        """Exponential Smoothing Forecast

        Args:
            series(pandas data frame) -- time series to perform forecast on
            alpha: smoothing factor
            n_periods: number of periods to do ses_f on
        """
        assert (type(series).__name__ == "Series")
        expS1 = list(series.index)
        result = [series[0]]
        for n in range(len(series) + n_periods):
            result.append(alpha * series[n] + (1 - alpha) * result[n - 1])

        forecast_date = Utilities.find_next_forecastdates(series, n_periods)
        expS1.append(forecast_date)
        return pd.Series(result, index=expS1)

    @staticmethod
    def holts_linear_f(series, alpha, beta, n_periods=1, mult_or_add=True):
        """Holts Linear Trend Forecast or Double Exponential Smoothing Forecast

        Args:
            series: pandas series object
            alpha: level smoothing factor
            beta: trend speed control factor
            mult_or_add: if True perform holts multiplicative smoothing,
                         else perform holts additive smoothing
        """
        expS1 = list(series.index)
        result = [series[0]]
        for n in range(1, len(series) + n_periods):
            if n == 1:
                level, trend = series[0], series[1] - series[0]
            if n >= len(series):
                value = result[-1]
            else:
                value = series[n]
            last_level, level = level, alpha * value + (1 - alpha) * (level +
                                                                      trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            if (mult_or_add):
                if (n <= len(series)):
                    result.append(level * trend)
                else:
                    result.append(level * trend * (n - len(series)))
            else:
                if (n <= len(series)):
                    result.append(level * trend)
                else:
                    result.append(level + (trend * (n - len(series))))
        forecast_dates = Utilities.find_next_forecastdates(series, n_periods)
        expS1.append(forecast_date)
        return pd.Series(result, index=expS1)

    @staticmethod
    def holt_winters_f(series, slen, alpha, beta, gamma, n_preds):
        """ holt winters additive seasonal forecast
        returns fully smoothed exponential forecast
        #TODO
        could possibly return timeseries + forecasted values
        """

        expS1 = list(series.index)
        result = []
        seasonals = Utilities.initial_seasonal_components(series, slen)
        for i in range(len(series) + n_preds):
            if i == 0:
                smooth = series[0]
                trend = Utilities.initial_trend(series, slen)
                result.append(series[0])
            elif i >= len(series):
                m = i - len(series) + 1
                result.append((smooth + m * trend) + seasonals[i % slen])
            else:
                val = series[i]
                last_smooth, smooth = smooth, alpha * (
                    val - seasonals[i % slen]) + (1 - alpha) * (smooth + trend)
                trend = beta * (smooth - last_smooth) + (1 - beta) * trend
                seasonals[i % slen] = gamma * (val - smooth) + (
                    1 - gamma) * seasonals[i % slen]
                result.append(smooth + trend + seasonals[i % slen])
        forecast_dates = Utilities.find_next_forecastdates(series, n_periods)
        expS1.append(forecast_date)
        return pd.Series(result, index=expS1)
    
    @staticmethod
    def theta_f(series, alpha, n_periods=1):
        # if seasonal, decompose and adjust for seasonality

        1) #TODO test for seasonality, if so deseasonalize. If so, remove seasonal factor, store
        n = len(series)
        forc = Statistical.ses_f(series, alpha, n_periods)
        ts_vals = series.to_numpy()
        b = linregress(np.arange(len(ts_vals), ts_vals))[0]
        drift = (0.5 * b) * (np.arange(n_periods) + 1/alpha - ((1-alpha)**n)/alpha)
        forc.array[-n_periods:] += drift

        # if seasonalized, reseasonalize the forecasted values
        return forc

