import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from dataset import PatternFrequencyDataset_top
from torch.utils.data import DataLoader
from datetime import datetime
import json
import torch.optim as optim


target_list = [ i for i in range(625)]
target_list.append(630)
# train the topology prediction model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_top(args, model, optimizer):
    seed = args.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    accuracies = {'all':[],'AB':[],'AC':[],'AD':[]}
    # Losses
    criterion = nn.CrossEntropyLoss()

    start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    df = pd.read_csv(args.train_dir)
    df = df.iloc[:,target_list]
    # training

    trainloader = DataLoader(PatternFrequencyDataset_top(df), batch_size=args.batch_size, shuffle=True, num_workers=4)
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        model.train()

        train_iter = iter(trainloader)
        for i in range(len(train_iter)):
            x, target = next(train_iter)
            x = x.to(device).float()
            target = target.to(device).view(-1)
            target = target.to(torch.long)
            y = model(x)
            loss = criterion(y, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # compute the accuracy on validation set

        temp_accuracies = evaluate(args,model)
        for k in temp_accuracies:
            accuracies[k].append(temp_accuracies[k])
    end_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # save the model
    torch.save({
        'start':start_time,
        'end':end_time,
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "model/"+args.net_name + ".pth")
    # save the validation accuracies
    a_file = open("accuracy"+args.net_name+"_epoch_" + str(args.epochs+args.restore_epoch)+".json", "w")
    json.dump(accuracies, a_file)
    a_file.close()

# accuracy of NN model
def accuracy(model,X,Y):
    predicted = model(X)
    pred_arrray = predicted.cpu().detach().numpy()
    pred_arrray = np.argmax(pred_arrray,axis = 1)
    Y_array = Y.cpu().detach().numpy()
    Y_array = Y_array.reshape(-1)
    msk = np.where(pred_arrray == Y_array)
    correct_pred = Y_array[msk]

    # AB type
    correct_AB = np.sum((correct_pred == 0))
    sum_AB = np.sum((Y_array == 0))

    # AC type
    correct_AC = np.sum((correct_pred == 1))
    sum_AC = np.sum((Y_array == 1))

    # AD type
    correct_AD = np.sum((correct_pred == 2))
    sum_AD = np.sum((Y_array == 2))

    return correct_AB,sum_AB,correct_AC,sum_AC,correct_AD,sum_AD

# compute the total accuracy on validation set
def evaluate(args,model):
    model.eval()
    df = pd.read_csv(args.validation_dir)
    df = df.iloc[:,target_list]
    valloader = DataLoader(PatternFrequencyDataset_top(df),batch_size=args.batch_size, shuffle=False)
    val_iter = iter(valloader)
    correct_AB = 0
    sum_AB = 0
    correct_AC = 0
    sum_AC = 0
    correct_AD = 0
    sum_AD = 0
    for i in range(len(val_iter)):
        x, target = next(val_iter)
        x = x.to(device).float()
        target = target.to(device).float().view(-1, 1)
        a, b, c, d, e, f = accuracy(model, x, target)

        correct_AB += a
        sum_AB += b
        correct_AC += c
        sum_AC += d
        correct_AD += e
        sum_AD += f

    accuracies = {'all': (correct_AB + correct_AC+correct_AD) / (sum_AB + sum_AC + sum_AD) * 100 if (sum_AB + sum_AC + sum_AD) else 0,
                  'AB': correct_AB / sum_AB * 100 if sum_AB > 0 else sum_AB,
                  'AC': correct_AC / sum_AC * 100 if sum_AC > 0 else sum_AC,
                  'AD': correct_AD / sum_AD * 100 if sum_AD > 0 else sum_AD}
    print(correct_AB , correct_AC,correct_AD)
    print(sum_AB , sum_AC , sum_AD)

    return accuracies
