# Libraries
import math
from sklearn.metrics import mean_squared_error as sk_mse

import math

import numpy as np


class Metrics:
    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        """Mean absolute percentage error.

        Args:
            y_true -- truth value
            y_pred -- Predicted value
            n -- number of periods
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        result = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return result

    @staticmethod
    def root_mean_square_error(y_true, y_pred, n=0):
        """Root mean squared error.

        Args:
            y_true -- truth value
            y_pred -- Predicted value
            n -- number of periods
        """
        if n == 0:
            # This is directly from the sklearn metric
            mse = sk_mse(y_true, y_pred)
            result = math.sqrt(mse)
            return result
        else:
            # This is directly from the sklearn metric
            mse = sk_mse(y_true, y_pred, n)
            result = math.sqrt(mse)
            return result

    # TODO Understand why the need is there to overload this method?s
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """Root mean squared error (Overloaded from sklearn).

        Args:
            y_true -- truth value
            y_pred -- Predicted value
            n -- number of periods
        """
        mse = sk_mse(y_true, y_pred)
        return mse

    @staticmethod
    def median_relative_absolute_error(y_true, y_pred, naive):
        """Median RAE (Overloaded from sklearn).

        Args:
            y_true -- truth value
            y_pred -- Predicted value
            n -- number of periods
        """
        result = []
        for indx in range(len(y_pred)):
            ii = (np.abs(np.array(y_true) - np.array(y_pred)) /
                  np.abs(np.array(naive) - np.array(y_true)))
            if ii is None:
                result.append(1)
            else:
                result.append(ii)
        return np.ma.median(result)
