import pandas as pd
import numpy as np
from dataset import PatternFrequencyDataset_top
from quartet_net import Quartet_Net_top
import torch
from torch.utils.data import DataLoader
import argparse

target_list = [ i for i in range(625)]
target_list.append(630)

def accuracy(model,X,Y):
    predicted = model(X)
    # print(predicted)
    # print(predicted)
    pred_arrray = predicted.cpu().detach().numpy()
    # print(pred_arrray.shape)
    pred_arrray = np.argmax(pred_arrray,axis = 1)
    return pred_arrray


def evaluate(model,df,batch_size):
    model.eval()
    df = df.iloc[:,target_list]
    valloader = DataLoader(PatternFrequencyDataset_top(df),batch_size=batch_size, shuffle=False)
    val_iter = iter(valloader)
    # correct_AB = 0
    # sum_AB = 0
    # correct_AC = 0
    # sum_AC = 0
    # correct_AD = 0
    # sum_AD = 0
    ls = []
    for i in range(len(val_iter)):
        x, target = next(val_iter)
        x = x.to(device).float()
        target = target.to(device).float().view(-1, 1)
        pred = accuracy(model, x, target)
        # print(pred)
        # print(target)
        # break
        ls.extend(pred)
        # a, b, c, d, e, f = accuracy(model, x, target)


    return ls
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', type=str, default='iq_net_top',help = 'name of the classifier network', required=False)
    parser.add_argument('--test_dir', type=str, default='./data/data_test_v3.csv', required=False)
    parser.add_argument('--output_dir', type=str, default='./iq_net_top_df.csv', required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    resume_dir = './model/' +args.net_name + '.pth'
    df = pd.read_csv(args.test_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Quartet_Net_top().to(device)
    model = torch.nn.DataParallel(model)
    batch_size = 64
    checkpoint = torch.load(resume_dir,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])


    ls = evaluate(model,df,batch_size)
    quartet_df = pd.DataFrame(ls,columns = ['pred_top'])
    quartet_df = pd.concat([quartet_df,df[['int_b','top','seq_length','tree_id']]],axis = 1)
    print((quartet_df['pred_top'] == quartet_df['top']).sum()/quartet_df.shape[0])
    quartet_df.to_csv(args.output_dir,index = False)