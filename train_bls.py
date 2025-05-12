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
# del target_list[624]
# train the branch length prediction model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_bls(args, model, optimizer):
    seed = args.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


    # torch.backends.cudnn.deterministic = True
    error = {'int_b':[],'A':[],'B':[],'C':[],'D':[]}
    # Losses
    # criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    # criterion = LogCoshMRELoss(alpha = args.alpha)
    criterion = LogCoshLoss()
    # criterion = HuberMRELoss(delta=0.4)

    start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    df = pd.read_csv(args.train_dir)
    df = df.iloc[:,target_list]
    # training
    trainloader = DataLoader(PatternFrequencyDataset_bls(df), batch_size=args.batch_size, shuffle=True, num_workers=4)
    temp_error = evaluate(args, model)
    print(temp_error)
    for epoch in range(args.epochs):


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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        # compute the accuracy on validation set
        temp_error = evaluate(args,model)
        print(temp_error)
        for k in temp_error.keys():
            error[k].append(temp_error[k])
    end_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # save the model
    torch.save({
        'start':start_time,
        'end':end_time,
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "model/"+args.net_name+".pth")
    # save the validation accuracies
    a_file = open("loss"+args.net_name+"_epoch_" + str(args.epochs+args.restore_epoch)+".json", "w")
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
    for i in range(len(val_iter)):
        x, target = next(val_iter)
        x = x.to(device).float()
        target = target.to(device).float().view(-1, 5)
        diff = sum_loss(model, x, target)
        sum_error = sum_error+ diff
    temp_error = {'int_b': sum_error[4]/N,
                  'A': sum_error[0]/N,
                  'B': sum_error[1]/N,
                  'C': sum_error[2]/N,
                  'D': sum_error[3]/N
                  }
    return temp_error
