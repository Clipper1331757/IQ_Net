import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import uniform, randint
import pandas as pd
# from quartet_net import  Quartet_Net_top
from iq_net import IQ_Net_top
import random
import torch

import numpy as np
from dataset import PatternFrequencyDataset_top, PatternFrequencyDataset_bls
from torch.utils.data import DataLoader

import json
import torch.optim as optim
import time
import datetime
import logging
# import psutil

device = 'cuda' if torch.cuda.is_available() else 'cpu'

target_list = [i for i in range(625)]
target_list.append(630)

seed = 757
np.random.seed(seed)
logging.basicConfig(
    filename="optuna_tune_qnet_top_log.txt",  # name
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",  # format
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_data_loaders(df_train, df_val, batch_size):
    num_workers = max(4, int(os.cpu_count() / torch.cuda.device_count()))
    train_loader = DataLoader(PatternFrequencyDataset_top(df_train),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(PatternFrequencyDataset_top(df_val),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True)
    return train_loader, val_loader

def objective(trial, df_train, df_val):
    start_time = time.time()

    # log_system_info()
    # hyperparameters

    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    # batch_size = trial.suggest_categorical("batch_size", [8,16,32, 64, 128,256,512])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)
    beta_1 = trial.suggest_float("beta_1", 0.85, 0.95)
    beta_2 = trial.suggest_float("beta_2", 0.9, 0.999)
    # weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    weight_decay = 0
    lr_decay = trial.suggest_float("lr_decay", 0.85, 1.0)

    if "batch_size" in trial.params:

        batch_size = trial.params["batch_size"]
    else:

        batch_size = trial.suggest_categorical(
            "batch_size_v2", [8, 16, 32, 64, 128, 256, 512]
        )
    trial.set_user_attr("batch_size", batch_size)
    params_for_log = {**trial.params, "batch_size": batch_size}
    logging.info(f"Start Trial {trial.number}: {params_for_log}")

    # logging.info(f"Start Trial {trial.number}: {trial.params}")

    train_loader, val_loader = get_data_loaders( df_train, df_val,batch_size)

    # load model
    model = IQ_Net_top(dropout_rate=dropout_rate)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta_1, beta_2), weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    # print('done')

    # train the model
    model.train()
    for epoch in range(10):
        # print(epoch)
        train_iter = iter(train_loader)
        for i in range(len(train_iter)):
            x, target = next(train_iter)

            x = x.to(device).float()
            target = target.to(device).view(-1)
            target = target.to(torch.long)

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
        # print(epoch)

    # validate
    model.eval()
    total_loss = 0.0
    total_samples = 0

    val_iter = iter(val_loader)
    with torch.no_grad():
        for i in range(len(val_iter)):
            x, target = next(val_iter)
            x = x.to(device).float()
            target = target.to(device).view(-1)
            target = target.to(torch.long)
            predicted = model(x)

            # pred_arrray = predicted.cpu().detach().numpy()
            #
            # pred_arrray = np.argmax(pred_arrray, axis=1)
            loss = criterion(predicted, target)
            bs = x.size(0)
            total_loss += loss.item() * bs  # sum of loss over this batch
            total_samples += bs
    # print(total_loss)
    avg_loss = total_loss / total_samples  # mean loss per sample
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"End Trial {trial.number}, loss: {avg_loss:.6f}, run time: {elapsed_time:.2f} s")
    return avg_loss


def main():
    start_time = datetime.datetime.now()
    logging.info(f"Start time: {start_time}")

    df_train = pd.read_csv('./data/df_train.csv')
    df_val = pd.read_csv('./data/df_val.csv')

    df_train = df_train.iloc[:, target_list]
    df_val = df_val.iloc[:, target_list]

    storage = "sqlite:///iq_net_top_tuning.db"
    study_name = "iq_net_top_tuning"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, df_train, df_val), n_trials=15)

    # print("Best hyperparameters:", study.best_params)

    best_params = study.best_params
    best_value = study.best_value

    print("Best hyperparameters:", best_params)

    # save best parameters
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    logging.info(f"End time: {end_time}")
    logging.info(f"Run time: {duration:.2f} s")
    logging.info(f"Best params: {best_params}, Best loss: {best_value:.6f}")


if __name__ == '__main__':

    from multiprocessing import freeze_support
    freeze_support()

    main()