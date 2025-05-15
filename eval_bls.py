import pandas as pd
import numpy as np
from dataset import PatternFrequencyDataset_top,PatternFrequencyDataset_bls
from quartet_net import Quartet_Net_bls
import argparse
import torch
from torch.utils.data import DataLoader

target_list = [ i for i in range(630)]

def sum_loss(predicted,Y):


    pred_arrray = predicted.cpu().detach().numpy()

    Y_array = Y.cpu().detach().numpy()

    # compute sum of the square error
    diff = Y_array - pred_arrray
    diff = np.abs(diff)
    diff = np.sum(diff,axis = 0)
    return diff.reshape(-1)


# compute the total accuracy on validation set
def evaluate(model,df,batch_size):
    model.eval()
    df = df.iloc[:, target_list]
    N = df.shape[0]
    valloader = DataLoader(PatternFrequencyDataset_bls(df),batch_size=batch_size, shuffle=False)
    val_iter = iter(valloader)
    sum_error = np.zeros(5)
    ls = None
    for i in range(len(val_iter)):
        x, target = next(val_iter)
        x = x.to(device).float()
        target = target.to(device).float().view(-1, 5)
        predicted = model(x)
        if not ls is None:
            ls = torch.cat((ls, predicted), dim=0)
        else:
            ls = predicted
        diff = sum_loss(predicted, target)
        sum_error = sum_error+ diff



    temp_error = {'int_b': sum_error[4]/N,
                  'A': sum_error[0]/N,
                  'B': sum_error[1]/N,
                  'C': sum_error[2]/N,
                  'D': sum_error[3]/N
                  }
    return ls.cpu().detach().numpy(),temp_error

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', type=str, default='iq_net_bls',help = 'name of the classifier network', required=False)
    parser.add_argument('--test_dir', type=str, default='./data/data_test_v3.csv', required=False)
    parser.add_argument('--output_dir', type=str, default='./iq_net_bls_df.csv', required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    resume_dir = './model/' +args.net_name + '.pth'
    df = pd.read_csv(args.test_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Quartet_Net_bls().to(device)

    model = torch.nn.DataParallel(model)

    batch_size = 64
    checkpoint = torch.load(resume_dir, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    ls, temp_error = evaluate(model, df, batch_size)
    quartet_df = pd.DataFrame(ls, columns=['pred_a', 'pred_b', 'pred_c', 'pred_d', 'pred_int_b'])
    quartet_df = pd.concat([quartet_df, df[['ext_b1', 'ext_b2', 'ext_b3', 'ext_b4', 'int_b', 'seq_length', 'tree_id']]],
                           axis=1)
    quartet_df.to_csv(args.output_dir, index=False)
