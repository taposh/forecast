"""Class docstring goes here.

"""
import math

from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from scipy.signal import periodogram as spectral_density
from scipy.stats import norm
from scipy.stats import linregress
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.tsatools import detrend
from statsmodels.tsa.stattools import acf

from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.tsa.filters._utils import _maybe_get_pandas_wrapper_freq
import statsmodels.api as sm


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

    @staticmethod
    def check_seasonality(x):
        ts_vals = x.to_numpy()
        n = len(ts_vals)
        m = int(Utilities.find_frequency(x))
        if (m > 1 and len(x) > 1 and n > m):
            r = acf(ts_vals, fft=True)[1:][:m]
            r_adj = np.delete(r, m - 1)
            stat = np.sqrt((1 + 2 * sum(r_adj**2)) / n)
            seasonal = (abs(r[m - 1]) / stat) > norm.ppf(0.95)
        else:
            seasonal = False
        return seasonal, m

    @staticmethod
    def decompose(df, period=365, lo_frac=0.6, lo_delta=0.01):
        """Create a seasonal-trend (with Loess, aka "STL") decomposition of 
        observed time series data.This implementation is modeled after the 
        ``statsmodels.tsa.seasonal_decompose`` method but substitutes a 
        Lowess regression for a convolution in its trend estimation.
        This is an additive model, Y[t] = T[t] + S[t] + e[t]        
        For more details on lo_frac and lo_delta, see: 
        `statsmodels.nonparametric.smoothers_lowess.lowess()`
        Args:
            df (pandas.Dataframe): Time series of observed counts.
                This DataFrame must be continuous 
                (no gaps or missing data), 
                and include a ``pandas.DatetimeIndex``.  
            period (int, optional): Most significant periodicity in 
                the observed time series, in units of 1 observation. 
                Ex: to accomodate strong annual periodicity within 
                years of daily observations, ``period=365``. 
            lo_frac (float, optional): Fraction of data to use in fitting Lowess regression. 
            lo_delta (float, optional): Fractional distance within which to use linear-interpolation 
                instead of weighted regression. Using non-zero ``lo_delta`` significantly decreases 
                computation time.
        Returns:
            `statsmodels.tsa.seasonal.DecomposeResult`: An object with DataFrame attributes for the 
                seasonal, trend, and residual components, as well as the average seasonal cycle. 
        """
        # use some existing pieces of statsmodels
        lowess = sm.nonparametric.lowess
        _pandas_wrapper, _ = _maybe_get_pandas_wrapper_freq(df)

        # get plain np array
        observed = np.asanyarray(df).squeeze()

        # calc trend, remove from observation
        trend = lowess(observed, [x for x in range(len(observed))],
                       frac=lo_frac,
                       delta=lo_delta * len(observed),
                       return_sorted=False)
        detrended = observed - trend

        # period must not be larger than size of series to avoid introducing NaNs
        period = min(period, len(observed))

        # calc one-period seasonality, remove tiled array from detrended
        period_averages = np.array(
            [pd_nanmean(detrended[i::period]) for i in range(period)])
        # 0-center the period avgs
        period_averages -= np.mean(period_averages)
        seasonal = np.tile(period_averages,
                           len(observed) // period + 1)[:len(observed)]
        resid = detrended - seasonal

        # convert the arrays back to appropriate dataframes, stuff them back into
        #  the statsmodel object
        results = list(map(_pandas_wrapper,
                           [seasonal, trend, resid, observed]))
        dr = DecomposeResult(seasonal=results[0],
                             trend=results[1],
                             resid=results[2],
                             observed=results[3],
                             period_averages=period_averages)
        return dr

    @staticmethod
    def find_frequency(x):
        """
        TODO fit auto regressive model to residuals. Get fitted values return
        """
        ts_vals = x.to_numpy()
        ts_vals = np.array(ts_vals)
        n = len(x)
        detrended_series = detrend(ts_vals)
        slope, intercept, _, _, _ = linregress(range(len(detrended_series)),
                                               detrended_series)
        residuals = np.array([
            detrended_series[idx] - ((slope * idx) - intercept)
            for idx, val in enumerate(detrended_series)
        ])
        f, spec = spectral_density(residuals)
        if (max(spec) > 3):
            max_freq_idx = np.argmax(spec)
            period = np.floor((1 / f[max_freq_idx]) + 0.5)
            if (period == np.inf):
                diff = [spec[i] - spec[i - 1] for i in range(1, len(spec))]
                j = []
                for idx, val in enumerate(diff):
                    if val > 0:
                        j.append(idx)
                if len(j):
                    next_max = j[0] + np.argmax(spec[j[0] + 1:])
                    if next_max < len(f):
                        period = np.floor((1 / f[next_max]) + 0.5)
                    else:
                        period = 1.0
                else:
                    period = 1.0
        else:
            period = 1.0

        return period
