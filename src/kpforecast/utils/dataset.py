import numpy as np
from torch.utils.data import Dataset


class DatasetTS(Dataset):
    """ Data Set Utility for Time Series.
    
        Args:
            - time_series(numpy 1d array) - array with univariate time series
            - forecast_length(int) - length of forecast window
            - backcast_length(int) - length of backcast window
            - sliding_window_coef(int) - determines how much to adjust sliding window
                by when determining forecast backcast pairs:
                    if sliding_window_coef = 1, this will make it so that backcast 
                    windows will be sampled that don't overlap. 
                    If sliding_window_coef=2, then backcast windows will overlap 
                    by 1/2 of their data. This creates a dataset with more training
                    samples, but can potentially lead to overfitting.
    """
    def __init__(self, time_series, forecast_length, backcast_length, sliding_window_coef=1):
        self.data = time_series
        self.forecast_length, self.backcast_length = forecast_length, backcast_length
        self.sliding_window_coef = sliding_window_coef
        self.sliding_window = int(np.ceil(self.backcast_length / sliding_window_coef))
    
    def __len__(self):
        """ Return the number of backcast/forecast pairs in the dataset.
        """
        length = int(np.floor((len(self.data)-(self.forecast_length+self.backcast_length)) / self.sliding_window))
        return length

    def __getitem__(self, index):
        """Get a single forecast/backcast pair by index.
            
            Args:
                index(int) - index of forecast/backcast pair
            raise exception if the index is greater than DatasetTS.__len__()
        """
        if(index > self.__len__()):
            raise IndexError("Index out of Bounds")
        # index = index * self.backcast_length
        index = index * self.sliding_window
        if index+self.backcast_length:
            backcast_model_input = self.data[index:index+self.backcast_length]
        else: 
            backcast_model_input = self.data[index:]
        forecast_actuals_idx = index+self.backcast_length
        forecast_actuals_output = self.data[forecast_actuals_idx:
                                            forecast_actuals_idx+self.forecast_length]
        forecast_actuals_output = np.array(forecast_actuals_output, dtype=np.float32)
        backcast_model_input = np.array(backcast_model_input, dtype=np.float32)
        return backcast_model_input, forecast_actuals_output


def data_generator(num_samples, backcast_length, forecast_length, signal_type='seasonality', random=False):
    def get_x_y():
        lin_space = np.linspace(-backcast_length, forecast_length, backcast_length)
        if random:
            offset = np.random.standard_normal() * 0.1
        else:
            offset = 1
        if signal_type == 'trend':
            x = lin_space + offset
        elif signal_type == 'seasonality':
            x = np.cos(2 * np.random.randint(low=1, high=3) * np.pi * lin_space)
            x += np.cos(2 * np.random.randint(low=2, high=4) * np.pi * lin_space)
            x += lin_space * offset + np.random.rand() * 0.1
        elif signal_type == 'cos':
            x = np.cos(2 * np.pi * lin_space)
        else:
            raise Exception('Unknown signal type.')
        x -= np.minimum(np.min(x), 0)
        x /= np.max(np.abs(x))
        return x

    while True:
        X = []
        for i in range(num_samples):
            x = get_x_y()
            X.append(x)
        yield np.array(X).flatten()
