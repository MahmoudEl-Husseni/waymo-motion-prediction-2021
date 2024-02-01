import config
from loss import *
from visualize import *
from train import Model, WaymoLoader
from ckpt_utils import load_checkpoint

import os
import cv2
import time
import logging
import argparse
import numpy as np
import pandas as pd
# import seaborn as sns

from tqdm import tqdm


# import plotly.graph_objects as go
# import plotly.io as pio



import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--model', type=str, default='model.pth', help='path to model')
    parser.add_argument('--train_data', type=str, default='data', help='path to train data')
    parser.add_argument('--val_data', type=str, default='data', help='path to val data')
    parser.add_argument('--save', type=str, default='save', help='Saving Path')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    args = parser.parse_args()
    return args


def forward(model, dataloader): 
    for i, data in enumerate(dataloader):
        x, y, is_avail = data
        _ = model(x)


def main():
    args = parse_arguments()

    # Load best model
    model_name = config.MODEL_NAME
    time_limit = config.FUTURE_TS
    n_traj = config.N_TRAJ
    model = Model(model_name, in_channels=config.IN_CHANNELS, time_limit=time_limit, n_traj=n_traj)

    model = nn.DataParallel(model)
    model.cuda()

    lr = config.LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("Loading model from last checkpoint")
    end_epoch = load_checkpoint(args.model, model, optimizer)

    dataset = WaymoLoader(args.train_data, limit=1000, return_vector=False)

    batch_size = config.TRAIN_BS
    num_workers = config.N_WORKERS
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=num_workers,
        pin_memory=False,
        # persistent_workers=True,
    )

    val_dataset = WaymoLoader(args.val_data, limit=1000, return_vector=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        # num_workers=num_workers,
        pin_memory=False,
        # persistent_workers=True,
    )

    # Train metrics
    y_true = np.empty((0, 80, 2))
    is_available = np.empty((0, 80))
    y_pred = np.empty((0, 6, 80, 2))
    confidence = np.empty((0, 6))

    for i, data in tqdm(enumerate(dataloader)):
        x, y, is_avail = data
        conf, logits = model(x)
        conf = conf.to('cpu').detach().numpy()
        logits = logits.to('cpu').detach().numpy()

        y_true = np.vstack((y_true, y))
        is_available = np.vstack((is_available, is_avail))
        y_pred = np.vstack((y_pred, logits))
        confidence = np.vstack((confidence, conf))
        # if i==10:
        #     break

    train_mde = mean_displacement_error(y_true, y_pred, is_available.reshape(-1, 80, 1))
    train_fde = final_displacement_error(y_true, y_pred, is_available.reshape(-1, 80, 1))
    train_mr_avg = missrate(y_true, y_pred, 0, comb='avg')
    train_mr_min = missrate(y_true, y_pred, 0, comb='min')
    mr_avg_df = pd.DataFrame(data=train_mr_avg, columns=[f't{i}' for i in range(80)])
    mr_avg_df['T'] = np.mean(train_mr_avg, axis=1)
    mr_avg_df.to_csv(args.save + '/train_miss_avg.csv')

    mr_min_df = pd.DataFrame(data=train_mr_min, columns=[f't{i}' for i in range(80)])
    mr_min_df['T'] = np.mean(train_mr_min, axis=1)
    mr_min_df.to_csv(args.save + '/train_miss_min.csv')


    # Val metrics
    y_true = np.empty((0, 80, 2))
    is_available = np.empty((0, 80))
    y_pred = np.empty((0, 6, 80, 2))
    confidence = np.empty((0, 6))

    for i, data in tqdm(enumerate(val_dataloader)):
        x, y, is_avail = data
        conf, logits = model(x)

        y_true = np.vstack((y_true, y))
        is_available = np.vstack((is_available, is_avail))
        y_pred = np.vstack((y_pred, logits.to('cpu').detach().numpy()))
        confidence = np.vstack((confidence, conf.to('cpu').detach().numpy()))
        # if i==10:
        #     break

    val_mde = mean_displacement_error(y_true, y_pred, is_available.reshape(-1, 80, 1))
    val_fde = final_displacement_error(y_true, y_pred, is_available.reshape(-1, 80, 1))
    val_mr_avg = missrate(y_true, y_pred, 0, comb='avg')
    val_mr_min = missrate(y_true, y_pred, 0, comb='min')

    MDE_c = [f"mde_{i}" for i in range(6)]
    FDE_c = [f"fde_{i}" for i in range(6)]
    df = pd.DataFrame(data={}, columns=MDE_c + FDE_c, index=['train', 'val'])
    df.loc['train'] = train_mde + train_fde
    df.loc['val'] = val_mde + val_fde
    df.to_csv(args.save + '/metrics.csv')

    miss_avg_df = pd.DataFrame(data=val_mr_avg, columns=[f't{i}' for i in range(80)])
    miss_avg_df['T'] = np.mean(val_mr_avg, axis=1)
    miss_avg_df.to_csv(args.save + '/miss_avg.csv')

    miss_min_df = pd.DataFrame(data=val_mr_min, columns=[f't{i}' for i in range(80)])
    miss_min_df['T'] = np.mean(val_mr_min, axis=1)
    miss_min_df.to_csv(args.save + '/miss_min.csv')


if __name__ == '__main__':
    main()
