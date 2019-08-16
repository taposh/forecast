from kpforecast.utils import data_generator

import numpy as np

NUM_SAMPLES_TO_GENERATE = 5000
batch_size = 1 # can set to whatever, so long as less than NUM_SAMPLES_TO_GENERATE
               # outdated parameter.
data_gen = data_generator(batch_size, backcast_length, forecast_length,
                                signal_type='seasonality', random=True)
ts = []
for grad_step, x, in enumerate(data_gen):
    ts.append(x)
    num_samples += len(x)
    if num_samples >= NUM_SAMPLES_TO_GENERATE:
        break
ts = np.concatenate(ts)