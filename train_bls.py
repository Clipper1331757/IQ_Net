import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from dataset import PatternFrequencyDataset_top,PatternFrequencyDataset_bls
from torch.utils.data import DataLoader
from datetime import datetime
import json
import torch.optim as optim
from loss_function import HuberMRELoss, LogCoshMRELoss, LogCoshLoss
target_list = [ i for i in range(630)]

# train the branch length prediction model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_bls(args, model, optimizer):
    seed = args.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



    patience = getattr(args, "patience", 5)
    min_delta = getattr(args, "min_delta", 1e-5)
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_model_state = None
    best_optim_state = None



    error = {'int_b':[],'A':[],'B':[],'C':[],'D':[]}
    # Losses
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    # criterion = LogCoshMRELoss(alpha = args.alpha)
    # criterion = LogCoshLoss()
    # criterion = HuberMRELoss(delta=0.4)

    start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    df = pd.read_csv(args.train_dir)

    df = df.iloc[:,target_list]
    # training
    trainloader = DataLoader(PatternFrequencyDataset_bls(df), batch_size=args.batch_size, shuffle=True, num_workers=4)


    train_loss = []
    val_loss = []
    for epoch in range(args.epochs):

        total_train_loss = 0
        total_train_samples = 0
        torch.cuda.empty_cache()
        model.train()
        train_iter = iter(trainloader)
        for i in range(len(train_iter)):
            x, target = next(train_iter)

            x = x.to(device).float()

            target = target.to(device).view(-1,5)
            target = target.to(torch.float)


            y = model(x)



            loss = criterion(y, target)

            bs = x.size(0)
            total_train_loss += loss.item() * bs  # sum of loss over this batch
            total_train_samples += bs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        # compute the accuracy on training set
        avg_train_loss = total_train_loss / total_train_samples  # mean loss per sample
        train_loss.append(avg_train_loss)

        temp_error, avg_val_loss = evaluate(args,model)
        val_loss.append(avg_val_loss)

        for k in temp_error.keys():
            error[k].append(temp_error[k])

        if avg_val_loss < best_val - min_delta:
            best_val = avg_val_loss
            best_epoch = epoch
            bad_epochs = 0
            # save the model status and optimizer status
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_optim_state = {k: v for k, v in optimizer.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. "
                      f"Best epoch was {best_epoch + 1} with val_loss={best_val:.6f}.")
                # load the best parameter
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                if best_optim_state is not None:
                    optimizer.load_state_dict(best_optim_state)
                # save train and validation loss

                for k in error:
                    error[k] = error[k][:best_epoch + 1]
                break
    end_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # save the model

    torch.save({
        'start': start_time,
        'end': end_time,
        'epoch': best_epoch + 1,
        'best_epoch': best_epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "model/" + args.net_name + ".pth")

    a_file = open("loss"+args.net_name+"_epoch_" + str((best_epoch + 1) + args.restore_epoch)+".json", "w")
    json.dump(error, a_file)
    a_file.close()

# accuracy of NN model
def sum_loss(model,X,Y):
    predicted = model(X)

    pred_arrray = predicted.cpu().detach().numpy()

    Y_array = Y.cpu().detach().numpy()

    # compute sum of the square error
    diff = Y_array - pred_arrray

    diff = np.abs(diff)
    diff = np.sum(diff,axis = 0)
    return diff.reshape(-1)


# compute the total accuracy on validation set
def evaluate(args,model):
    model.eval()
    df = pd.read_csv(args.validation_dir)
    df = df.iloc[:,target_list]
    N = df.shape[0]
    valloader = DataLoader(PatternFrequencyDataset_bls(df),batch_size=args.batch_size, shuffle=False)
    val_iter = iter(valloader)
    sum_error = np.zeros(5)
    total_val_loss = 0
    total_val_samples = 0
    criterion = nn.MSELoss()
    for i in range(len(val_iter)):
        x, target = next(val_iter)
        x = x.to(device).float()
        target = target.to(device).float().view(-1, 5)
        diff = sum_loss(model, x, target)
        sum_error = sum_error+ diff

        target = target.to(torch.float)


        y = model(x)
        loss = criterion(y, target)

        bs = x.size(0)
        total_val_loss += loss.item() * bs  # sum of loss over this batch
        total_val_samples += bs

    avg_val_loss = total_val_loss / total_val_samples
    temp_error = {'int_b': sum_error[4]/N,
                      'A': sum_error[0]/N,
                      'B': sum_error[1]/N,
                      'C': sum_error[2]/N,
                      'D': sum_error[3]/N
                      }


    return temp_error,avg_val_loss
