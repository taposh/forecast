"""
Utilities for managing univariate time series.
"""
import numpy as np
import matplotlib.pyplot as plt

class TSUtils:

    def save_ts(ts, path):
        """ Saves a time series in numpy format, and as a graph.

        Args:
            - ts(1 dim ndarray): the time series to be saved
            - path(str): path to save time series and time series graph to.
        """
        if path[-1] != "/":
            path = path + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + "timeseries.npy", ts)
        plt.figure()
        x = range(len(ts))
        y = ts
        plt.plot(x, y)
        plt.xlabel("Time")
        plt.ylabel("Observation")
        plt.title(RUN_NAME + " Original Time Series")
        plt.savefig(path + "original_time_series.png")
        plt.cla()

    @staticmethod
    def min_max_normalize(ts):
        """ Perform minmax normalization on a timeseries.
        
        Args:
            - ts(1 dim ndarray): time series to be normalized.
        
        Returns:
            - normalized_ts (ndarray).
            - min_val(float) - min value of original time series.
            - max_val(float) - max value of original time series.
        """
        min_val = ts.min()
        max_val = ts.max()
        normalized_ts = [(x - min_val) / (max_val - min_val) for x in ts]
        return np.array(normalized_ts), min_val, max_val

    def min_max_unnormalize(ts, max_val, min_val):
        """ Unnormalize ts or single value that was normalized 
            by minmax normalization.
        
        Args:
            - ts(1dim ndarray or float): value or array of time series
                values to be unnormalized, assuming value(s) were 
                previously normalized by minmax normalization.
            - max_val(float) - max value of original time series
            - min_val(float) - min val of original time series

        Returns:
            - Unnormalized series/value
        """
        return (ts * (max_ts_val-min_ts_val)) + min_ts_val

