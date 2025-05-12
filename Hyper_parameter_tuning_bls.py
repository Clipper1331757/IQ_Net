import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import uniform, randint
import pandas as pd
from quartet_net import Quartet_Net_top,Quartet_Net_bls
import random
import torch

import numpy as np
from dataset import  PatternFrequencyDataset_bls
from torch.utils.data import DataLoader

import json
import torch.optim as optim
import time
import datetime
import logging
# import psutil

device = 'cuda' if torch.cuda.is_available() else 'cpu'
target_list = [ i for i in range(630)]

seed = 757
np.random.seed(seed)
# log info
logging.basicConfig(
    filename="optuna_tune_qnet_bls_log.txt",  # name
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",  # format
    datefmt="%Y-%m-%d %H:%M:%S",
)

# def log_system_info():
#     """record memory usage"""
#     process = psutil.Process()
#     mem_info = process.memory_info()
#     logging.info(f"memory used: {mem_info.rss / 1024 / 1024:.2f} MB")


def get_data_loaders(batch_size):

    # df_train = pd.read_csv('./data/data_train_v3.csv')
    # use small dataset just to test the code, directly run on the large train set take too much time.
    df_train = pd.read_csv('./data/data_val_v3.csv')
    df_val = pd.read_csv('./data/data_val_v3.csv')
    df_train = df_train.iloc[:, target_list]
    # print(df_train.shape)
    df_val = df_val.iloc[:, target_list]


    train_loader = DataLoader(PatternFrequencyDataset_bls(df_train), batch_size=batch_size, shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(PatternFrequencyDataset_bls(df_val), batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, val_loader

def objective(trial):
    start_time = time.time()
    logging.info(f"Start Trial {trial.number}: {trial.params}")

    # log_system_info()
    # hyperparameters

    lr = trial.suggest_float("lr", 1e-6, 1e-2,log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout_rate = trial.suggest_float("dropout_rate",0, 0.5)
    beta_1 = trial.suggest_float("beta_1",  0.5, 0.99)
    beta_2 = trial.suggest_float("beta_2",  0.5, 0.999)
    weight_decay = trial.suggest_float("weight_decay",1e-6, 0.01)
    lr_decay = trial.suggest_float("lr_decay", 0.8, 1)


    train_loader, val_loader = get_data_loaders(batch_size)

    # load model
    model = Quartet_Net_bls(dropout_rate=dropout_rate)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta_1, beta_2), weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)

    # train the model
    model.train()
    for epoch in range(1):
        # print(epoch)
        train_iter = iter(train_loader)
        for i in range(len(train_iter)):
            x, target = next(train_iter)

            x = x.to(device).float()
            target = target.to(device)
            target = target.to(torch.float)

            y = model(x)
            # print(y.shape)
            # print(y)
            # print(target.shape)
            # print(target)

            loss = criterion(y, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if i % 1000 == 0:
            #     print(i)
        scheduler.step()

    # validate
    model.eval()
    total_loss = 0.0

    val_iter = iter(val_loader)
    with torch.no_grad():
        for i in range(len(val_iter)):
            x, target = next(val_iter)
            x = x.to(device).float()
            target = target.to(device)
            target = target.to(torch.float)
            predicted = model(x)


            # compute loss
            loss = criterion(predicted, target)
            total_loss += loss.item()
    # print(total_loss)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"End Trial {trial.number}, loss: {loss:.6f}, run time: {elapsed_time:.2f} s")
    return total_loss


def main():
    start_time = datetime.datetime.now()
    logging.info(f"Start time: {start_time}")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)


    print("Best hyperparameters:", study.best_params)
    with open('bls_best_params_bls.json', 'w') as file:
        json.dump(study.best_params, file)
    end_time = datetime.datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    logging.info(f"End time: {end_time}")
    logging.info(f"Total run time: {total_duration:.2f} s")
    logging.info(f"best hyperparameter: {study.best_params}, best loss: {study.best_value:.6f}")

if __name__ == '__main__':

    from multiprocessing import freeze_support
    freeze_support()

    main()