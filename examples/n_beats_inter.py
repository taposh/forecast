
from kpforecast.ml import NBeats
from kpforecast.ml.n_beats.blocks import (GenericNBeatsBlock,
                                   TrendBlock,
                                   SeasonalityBlock)
from kpforecast.utils import DatasetTS, data_generator

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
from torch.utils.data import (BatchSampler, DataLoader,
                              SubsetRandomSampler)
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot, make_dot_from_trace

parser = ArgumentParser(description='N-Beats')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--disable-plot', action='store_true', help='Disable interactive plots')
parser.add_argument('--checkpoint', default="", help = "checkpoint dir")
parser.add_argument('--tensorboard', action="store_true", help="Enable Tensorboard")
args = parser.parse_args()
DEVICE = torch.device('cuda') if not args.disable_cuda and torch.cuda.is_available() else torch.device('cpu')
DISABLE_PLOT = args.disable_plot
CHECKPOINT_NAME = args.checkpoint
TENSORBOARD_ENABLE = args.tensorboard
if CHECKPOINT_NAME == "":
    RUN_NAME = str(datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))
else:
    RUN_NAME = CHECKPOINT_NAME[5:22]

def train():
    forecast_length = 3
    backcast_length = 24
    batch_size = 1  # greater than 4 for viz
    f_b_dim = (forecast_length, backcast_length)
    num_samples = 0
    lr = 1e-3

    # If the user specified a checkpoint, load the data
    # and model from that checkpoint/run
    if CHECKPOINT_NAME == "":
        data_gen = data_generator(batch_size, backcast_length, forecast_length,
                            signal_type='seasonality', random=True)
        ts = []
        # user did not specify a previous checkpoint, generate new data
        for grad_step, x, in enumerate(data_gen):
            ts.append(x)
            num_samples += len(x)
            if num_samples >= 100000:
                break
        ts = np.concatenate(ts)
        save_ts(ts)
    else:
        ts = np.load("data/" + RUN_NAME + "/dataset/timeseries.npy")
    data = DatasetTS(ts, forecast_length, backcast_length)
    net = NBeats(stacks=[TrendBlock, SeasonalityBlock, GenericNBeatsBlock],
                 f_b_dim=f_b_dim,
                 num_blocks_per_stack=4,
                 thetas_dims=[[2,2], [8,8], [2,8]],
                 hidden_layer_dim=64)
    # net = NBeats(stacks=[GenericNBeatsBlock] * 2,
    #              f_b_dim=f_b_dim,
    #              num_blocks_per_stack=4,
    #              thetas_dims=[[8,8], [8,8]],
    #              hidden_layer_dim=64)
    # net = NBeats(stacks=[SeasonalityBlock, GenericNBeatsBlock],
    #              f_b_dim=f_b_dim,
    #              num_blocks_per_stack=4,
    #              thetas_dims=[[8,8], [8,8]],
    #              hidden_layer_dim=128)


    optimiser = optim.Adam(net.parameters(), lr=lr)
    print('--- Training ---')
    epochs = 50
    initial_epoch_step = load_model(net, optimiser)
    ds_len = len(data)
    train_length = int(np.ceil(ds_len*0.95))
    train_sampler = SubsetRandomSampler(list(range(train_length)))
    validation_sampler = SubsetRandomSampler(list(range(train_length + 1, ds_len)))
    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=100,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(data,
                                                    batch_size=4,
                                                    sampler=validation_sampler)
    backcast_validation_data, target_validation = iter(validation_loader).next()
    writer = SummaryWriter("data/" + RUN_NAME) if TENSORBOARD_ENABLE else None
    for epoch in range(initial_epoch_step, epochs):
        for grad_step, (x, target) in enumerate(train_loader):
            optimiser.zero_grad()
            net.train()
            forecast, backcast = net(x)
            loss = F.smooth_l1_loss(forecast, target)
            temp = loss
            # loss = loss * 100
            #loss = F.smooth_l1_loss(forecast, target)
            loss.backward()
            optimiser.step()
        if writer:
            writer.add_scalar("loss/training_loss", temp, epoch)
        with torch.no_grad():
            # save_model(net, optimiser, epoch)
            if not DISABLE_PLOT:
                plot(net,
                     backcast_validation_data,
                     target_validation,
                     validation_loader,
                     backcast_length,
                     forecast_length,
                     epoch,
                     writer)
    plt.close()

def plot(net,
         backcast_validation_data,
         target_validation,
         loaded_dataset,
         backcast_length,
         forecast_length,
         epoch,
         writer=None):
    net.eval()
    # backcast_data, target = iter(loaded_dataset).next()
    backcast_data, target = backcast_validation_data, target_validation
    forecast_predictions, _ = net(backcast_data)
    fig = plt.figure(1)
    plt.subplots_adjust(top=0.88)
    assert len(backcast_data) == len(target) == len(forecast_predictions)
    num_validations = len(backcast_data)
    subplots = list(range(221, 221+num_validations))
    for i in range(num_validations):
        try:
            xx, yy = np.array(backcast_data[i]), np.array(target[i])
            ff = np.array(forecast_predictions[i])
        except:
            import ipdb; ipdb.set_trace()
        plt.subplot(subplots[i])
        plt.plot(range(backcast_length), xx, color='b')
        plt.plot(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plt.plot(range(backcast_length, backcast_length + forecast_length), ff, color='r')
        plt.title(f'step #{epoch} sample i ({i})')
    path = "./data/" + RUN_NAME + "/validation_graphs/"
    if not os.path.exists(path):
        os.makedirs(path)
    if not epoch%10:
        plt.savefig(path + "validation_epoch_" + str(epoch))
    if writer:
        writer.add_figure('validation/fitted_data', fig)
    validation_losses = []
    sMAPE = 0
    num_eval_samples = 0
    for idx, (x, target) in enumerate(loaded_dataset):
        forecasted_out, _ = net(x)
        for idx, (targ, forc) in enumerate(zip(target, forecasted_out)):
            num_eval_samples += 1
            sMAPE += np.abs(forc-targ) / ((np.abs(targ)+np.abs(forc))/2)
        smooth_l1_loss = F.smooth_l1_loss(target, forecasted_out)
        validation_losses.append(smooth_l1_loss)
    sMAPE = torch.mean(sMAPE / num_eval_samples)
    plt.clf()
    final_loss = np.mean(validation_losses)
    if writer:
        writer.add_scalar("loss/sMAPE", sMAPE, epoch)
        writer.add_scalar('loss/validation_loss', final_loss, epoch)
    return final_loss

def save_model(model, optimiser, epoch):
    path = "data/" + RUN_NAME + "/saved_models/"
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, path + "model_epoch_" + str(epoch) + ".th")

def load_model(model, optimiser):
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
        return epoch
    return 0

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

if __name__ == '__main__':
    train()
