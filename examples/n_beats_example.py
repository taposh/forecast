from kpforecast.ml import GenericNBeatsBlock, NBeats, Stack

import datetime
from argparse import ArgumentParser
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot, make_dot_from_trace

parser = ArgumentParser(description='N-Beats')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--disable-plot', action='store_true', help='Disable interactive plots')
parser.add_argument('--checkpoint', default="", help = "checkpoint dir")
args = parser.parse_args()
DEVICE = torch.device('cuda') if not args.disable_cuda and torch.cuda.is_available() else torch.device('cpu')
DISABLE_PLOT = args.disable_plot
CHECKPOINT_NAME = args.checkpoint
if CHECKPOINT_NAME == "":
    RUN_NAME = str(datetime.datetime.now().strftime("%H%M%S-%Y-%m-%d"))
else:
    RUN_NAME = CHECKPOINT_NAME[5:22]

class DatasetTS(Dataset):
    
    def __init__(self, time_series, forecast_length, backcast_length):
        self.data = time_series
        self.forecast_length, self.backcast_length = forecast_length, backcast_length
    def __len__(self):
        return len(self.data)-(self.forecast_length + self.backcast_length)
    
    def __getitem__(self, index):
        if(index > self.__len__()):
            raise IndexError("Index out of Bounds")
        if index+self.backcast_length:
            backcast_model_input = self.data[index:index+self.backcast_length]
        else: 
            backcast_model_input = self.data[index:]
        forecast_actuals_idx = index+self.backcast_length
        forecast_actuals_output = self.data[forecast_actuals_idx:
                                            forecast_actuals_idx+self.forecast_length]
        backcast_model_input = torch.tensor(backcast_model_input, dtype=torch.float)
        forecast_actuals_output = torch.tensor(forecast_actuals_output, dtype=torch.float)
        return backcast_model_input, forecast_actuals_output


def get_data(num_samples, backcast_length, forecast_length, signal_type='seasonality', random=False):
    def get_x_y():
        lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
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
        x = np.expand_dims(x, axis=0)
        y = x[:, backcast_length:]
        x = x[:, :backcast_length]
        return x[0]

    while True:
        X = []
        for i in range(num_samples):
            x = get_x_y()
            X.append(x)
        yield np.array(X).flatten()

def train():
    forecast_length = 10
    backcast_length = 5 * forecast_length
    batch_size = 100  # greater than 4 for viz
    f_b_dim = (forecast_length, backcast_length)
    num_samples = 0

    # If the user specified a checkpoint, load the data
    # and model from that checkpoint/run
    if CHECKPOINT_NAME == "":
        data_gen = get_data(batch_size, backcast_length, forecast_length,
                            signal_type='seasonality', random=True)
        ts = []
        # user did not specify a previous checkpoint, generate new data
        for grad_step, x, in enumerate(data_gen):
            ts.append(x)
            num_samples += len(x)
            if num_samples >= 10000:
                break
        ts = np.concatenate(ts)
        save_ts(ts)
    else:
        ts = np.load("data/" + RUN_NAME + "/dataset/timeseries.npy")
    data = DatasetTS(ts, forecast_length, backcast_length)

    print('--- Model ---')
    net = NBeats(stacks=[GenericNBeatsBlock, GenericNBeatsBlock],
                 f_b_dim=f_b_dim,
                 num_blocks_per_stack=3,
                 thetas_dim=[2,8],
                 hidden_layer_dim=1024)

    optimiser = optim.Adam(net.parameters())
    start_time = time()
    print('--- Training ---')
    initial_grad_step = load(net, optimiser)
    for grad_step, (x, target) in enumerate(data):
        # grad_step += initial_grad_step
        # optimiser.zero_grad()
        # net.train()
        # forecast, backcast = net(x)
        # loss = F.mse_loss(forecast, target)
        # loss.backward()
        # optimiser.step()
        # if grad_step % 1000 == 0 or (grad_step < 1000 and grad_step % 100 == 0):
        #     with torch.no_grad():
        #         elapsed = int(time() - start_time)
        #         print(f'{str(grad_step).zfill(6)} {loss.item():.6f} {elapsed}')
        #         save(net, optimiser, grad_step)
        #         if not DISABLE_PLOT:
        #             plot(net, x, target, backcast_length, forecast_length, grad_step)

def save_ts(ts):
    path = "data/" + RUN_NAME + "/dataset/"
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

def save(model, optimiser, grad_step):
    path = "data/" + RUN_NAME + "/saved_models/"
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, path + "model_" + str(grad_step) + ".th")

def load(model, optimiser):
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
        return grad_step
    return 0

def plot(net, x, target, backcast_length, forecast_length, grad_step):
    net.eval()
    f, _ = net(torch.tensor(x, dtype=torch.float))
    subplots = [221, 222, 223, 224]

    plt.figure(1)
    plt.subplots_adjust(top=0.88)
    for i in range(4):
        ff, xx, yy = f.cpu().numpy()[i], x[i], target[i]
        plt.subplot(subplots[i])
        plt.plot(range(0, backcast_length), xx, color='b')
        plt.plot(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plt.plot(range(backcast_length, backcast_length + forecast_length), ff, color='r')
        plt.title(f'step #{grad_step} ({i})')

    plt.show()

if __name__ == '__main__':
    train()
