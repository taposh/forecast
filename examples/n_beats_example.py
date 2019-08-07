from kpforecast.ml import GenericNBeatsBlock, NBeats, Stack

from torch.utils.data import DataLoader, Dataset

import os
from argparse import ArgumentParser
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot, make_dot_from_trace

# def get_data(num_samples, backcast_length, forecast_length, signal_type='seasonality', random=False):
#     def get_x_y():
#         lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
#         if random:
#             offset = np.random.standard_normal() * 0.1
#         else:
#             offset = 1
#         if signal_type == 'trend':
#             x = lin_space + offset
#         elif signal_type == 'seasonality':
#             x = np.cos(2 * np.random.randint(low=1, high=3) * np.pi * lin_space)
#             x += np.cos(2 * np.random.randint(low=2, high=4) * np.pi * lin_space)
#             x += lin_space * offset + np.random.rand() * 0.1
#         elif signal_type == 'cos':
#             x = np.cos(2 * np.pi * lin_space)
#         else:
#             raise Exception('Unknown signal type.')
#         x -= np.minimum(np.min(x), 0)
#         x /= np.max(np.abs(x))
#         x = np.expand_dims(x, axis=0)
#         y = x[:, backcast_length:]
#         x = x[:, :backcast_length]
#         return x[0], y[0]

#     X = []
#     Y = []
#     for i in range(num_samples):
#         x, y = get_x_y()
#         X.append(x)
#         Y.append(y)
#     return np.array(X).flatten()


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
        backcast_model_input = torch.from_numpy(backcast_model_input)
        forecast_actuals_output = torch.from_numpy(forecast_actuals_output)
        return backcast_model_input, forecast_actuals_output




# def train(forward_length, backwards_length, data, train_dict):
#     f_b_dim = ((forward_length, backwards_length))
#     DS = DatasetTS(data, f_b_dim[0], f_b_dim[1])

#     model = NBeats(stacks=[GenericNBeatsBlock, GenericNBeatsBlock],
#                 f_b_dim=f_b_dim,
#                 num_blocks_per_stack=2,
#                 thetas_dim=(2,8),
#                 hidden_layer_dim=64
#                 )
#     optimiser = optim.Adam(model.parameters(), lr=train_dict['lr'])
#     torch.autograd.set_detect_anomaly(True)
#     # writer = SummaryWriter()
#     # in place operations are causing to fail!
#     losses = []
#     for i in range(train_dict['num_itr']):
#         x, y = DS[i * 25]

#         model.train()
#         forecast, backcast = model(x)
#         try:
#             loss = F.mse_loss(forecast, y)
#         except:
#             import ipdb; ipdb.set_trace()
#         optimiser.zero_grad()
#         loss.backward()
#         optimiser.step()

#         losses.append(loss)
#         # writer.add_graph(model, x)
#         # writer.add_scalar("loss", loss)
#     # writer.close()
#     # start_prediction_idx = -(forecast_length+backwards_length)
#     back_cast_fin, _ = DS[-backwards_length]
#     prediction = model(back_cast_fin)
#     return prediction, losses





# if __name__ == "__main__":
#     torch.set_default_dtype(torch.float64)
#     data_gen_length = 10000
#     forecast_length = 10
#     backcast_length = 5*forecast_length
#     train_length = data_gen_length-forecast_length
#     data_gen_smoothing_param = 10 * forecast_length
#     batch_size = 10000
#     data_gen = get_data(batch_size, data_gen_smoothing_param, forecast_length,
#                         signal_type='seasonality', random=True)
#     data_gen = data_gen[:data_gen_length]
#     train_dict = {'lr': 0.001, 'num_itr': 200}
#     ts = data_gen
#     actuals = data_gen[train_length:data_gen_length]

#     prediction, losses = train(forecast_length, backcast_length, ts, train_dict)
#     # actuals
#     # plt.plot(range(train_length,data_gen_length), actuals, 'bo-')
#     # training
#     # plt.plot(range(len(ts)), ts, 'g--')
#     # predictions
#     prediction = prediction[0].detach().numpy()

#     # plt.plot(range(train_length,data_gen_length), prediction, 'ro-')
#     f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#     ax1.plot(range(train_length,data_gen_length), actuals, 'bo-')
#     ax1.plot(range(len(ts)), ts, 'g--')
#     ax1.plot(range(train_length,data_gen_length), prediction, 'ro-')

#     ax2.plot(range(len(losses)), losses)
#     print(F.mse_loss(torch.tensor(prediction), torch.tensor(actuals)))
#     plt.show()


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

'''
import os
from argparse import ArgumentParser
from time import time

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.nn import functional as F

from data import get_data
from model import NBeatsNet
''' 
parser = ArgumentParser(description='N-Beats')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--disable-plot', action='store_true', help='Disable interactive plots')
parser.add_argument('--checkpoint', default="", help = "checkpoint dir")
args = parser.parse_args()
DEVICE = torch.device('cuda') if not args.disable_cuda and torch.cuda.is_available() else torch.device('cpu')
DISABLE_PLOT = args.disable_plot
CHECKPOINT_NAME = args.checkpoint


def train():
    forecast_length = 10
    backcast_length = 5 * forecast_length
    batch_size = 100  # greater than 4 for viz
    f_b_dim = (forecast_length, backcast_length)
    num_samples = 0
    data_gen = get_data(batch_size, backcast_length, forecast_length,
                        signal_type='seasonality', random=True)
    ts = []
    for grad_step, x, in enumerate(data_gen):
        if num_samples == 5000:
            break
        ts.append(x)
        num_samples += len(x)
    ts = np.concatenate(ts)
    data = DatasetTS(ts, forecast_length, backcast_length)
    print('--- Model ---')
    # net = NBeatsNet(device=DEVICE,
    #                 stack_types=[NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK],
    #                 forecast_length=forecast_length,
    #                 thetas_dims=[2, 8],
    #                 nb_blocks_per_stack=3,
    #                 backcast_length=backcast_length,
    #                 hidden_layer_units=1024,
    #                 share_weights_in_stack=False)

    net = NBeats(stacks=[GenericNBeatsBlock, GenericNBeatsBlock],
                f_b_dim=f_b_dim,
                num_blocks_per_stack=3,
                thetas_dim=[2,8],
                hidden_layer_dim=1024
                )

    # net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
    #                 forecast_length=forecast_length,
    #                 thetas_dims=[7, 8],
    #                 nb_blocks_per_stack=3,
    #                 backcast_length=backcast_length,
    #                 hidden_layer_units=128,
    #                 share_weights_in_stack=False)

    optimiser = optim.Adam(net.parameters())
    start_time = time()

    print('--- Training ---')
    initial_grad_step = load(net, optimiser)
    for grad_step, (x, target) in enumerate(data):
        grad_step += initial_grad_step
        optimiser.zero_grad()
        net.train()
        forecast, backcast = net(torch.tensor(x, dtype=torch.float).to(DEVICE))
        loss = F.mse_loss(forecast, torch.tensor(target, dtype=torch.float).to(DEVICE))
        loss.backward()
        optimiser.step()
        if grad_step % 1000 == 0 or (grad_step < 1000 and grad_step % 100 == 0):
            with torch.no_grad():
                elapsed = int(time() - start_time)
                print(f'{str(grad_step).zfill(6)} {loss.item():.6f} {elapsed}')
                save(net, optimiser, grad_step)
                if not DISABLE_PLOT:
                    plot(net, x, target, backcast_length, forecast_length, grad_step)
        if grad_step > 10000:
            print('Finished.')
            break


def save(model, optimiser, grad_step):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, CHECKPOINT_NAME)


def load(model, optimiser):
    import ipdb; ipdb.set_trace()
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
