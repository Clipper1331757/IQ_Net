import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import uniform, randint
import pandas as pd
from iq_net import IQ_Net_bls
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

def get_data_loaders(df_train, df_val, batch_size):
    num_workers = max(4, int(os.cpu_count() / torch.cuda.device_count()))
    train_loader = DataLoader(PatternFrequencyDataset_bls(df_train),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(PatternFrequencyDataset_bls(df_val),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True)
    return train_loader, val_loader

def objective(trial, df_train, df_val):
    start_time = time.time()

    # log_system_info()
    # hyperparameters

    lr = trial.suggest_float("lr", 1e-5, 1e-2,log=True)
    # batch_size = trial.suggest_categorical("batch_size", [8,16,32, 64, 128,256,512])
    dropout_rate = trial.suggest_float("dropout_rate",0, 0.5)
    beta_1 = trial.suggest_float("beta_1",  0.9, 0.99)
    beta_2 = trial.suggest_float("beta_2",  0.9, 0.999)
    weight_decay = trial.suggest_float("weight_decay",1e-6, 1e-3, log = True)
    lr_decay = trial.suggest_float("lr_decay", 0.85, 1)


    if "batch_size" in trial.params:

        batch_size = trial.params["batch_size"]
    else:

        batch_size = trial.suggest_categorical(
            "batch_size_v2", [8, 16, 32, 64, 128, 256, 512]
        )

    # logging.info(f"Start Trial {trial.number}: {trial.params}")

    trial.set_user_attr("batch_size", batch_size)

    params_for_log = {**trial.params, "batch_size": batch_size}
    logging.info(f"Start Trial {trial.number}: {params_for_log}")

    # logging.info(f"Start Trial {trial.number}: {trial.params}")
    train_loader, val_loader = get_data_loaders( df_train, df_val,batch_size)

    # load model
    model = IQ_Net_bls(dropout_rate=dropout_rate)
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
    total_samples = 0

    with torch.no_grad():
        for x, target in val_loader:
            x = x.to(device)
            target = target.to(device)
            output = model(x)
            loss = criterion(output, target)

            bs = x.size(0)
            total_loss += loss.item() * bs  # sum of loss over this batch
            total_samples += bs

    avg_loss = total_loss / total_samples  # mean loss per sample
    # print(total_loss)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"End Trial {trial.number}, loss: {loss:.6f}, run time: {elapsed_time:.2f} s")
    return avg_loss


def main():

    start_time = datetime.datetime.now()
    logging.info(f"Start time: {start_time}")

    df_train = pd.read_csv('./data/df_train.csv')
    df_val = pd.read_csv('./data/df_val.csv')

    df_train = df_train.iloc[:, target_list]
    df_val = df_val.iloc[:, target_list]

    storage = "sqlite:///iq_net_bls_tuning.db"
    study_name = "iq_net_bls_tuning"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, df_train, df_val), n_trials=50)



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