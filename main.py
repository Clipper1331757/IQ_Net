import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
from train_top import train_top
import numpy as np
from train_bls import train_bls
from iq_net import IQ_Net_bls,IQ_Net_top

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', type=str, default='iq_net_top',help = 'name of the network', required=False)
    parser.add_argument('--type', type=str, default='top', help='type of the network, bls or top', required=False)
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model',required=False)
    parser.add_argument('--resume_dir', type=str, default='./model/quartet_net_bls_test_log_cosh.pth', help='dir of resumed model', required=False)
    parser.add_argument('--restore_epoch', type=int, default=0, help='restore epochs',required=False)
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs',required=False)
    parser.add_argument('--batch_size', type=int, default=64,required=False)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate',required=False)
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='learning rate', required=False)
    parser.add_argument('--beta_1', type=float, default=0.95, help='learning rate', required=False)
    parser.add_argument('--beta_2', type=float, default=0.99, help='learning rate', required=False)
    # parser.add_argument('--train_dir',type=str,default='./data/data_train_v2_final.csv',required=False)
    # parser.add_argument('--validation_dir',type=str,default='./data/data_val_v2_final.csv',required=False)

    parser.add_argument('--train_dir',type=str,default='./data/df_train.csv',required=False)
    # parser.add_argument('--train_dir', type=str, default='./data/data_val_v3.csv', required=False)
    parser.add_argument('--validation_dir',type=str,default='./data/df_val.csv',required=False)

    parser.add_argument('--seed', type=int, default=757, required=False)
    parser.add_argument('--lr_decay', type=float, default=0.86, required=False)
    parser.add_argument('--weight_decay', type=float, default=0, required=False)
    parser.add_argument('--alpha', type=float, default=0.9, help = 'alpha of combined MRE loss',required=False)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # network
    if args.type =='bls':
        model = IQ_Net_bls(dropout_rate=args.dropout_rate)
    else:
        model = IQ_Net_top(dropout_rate=args.dropout_rate)

    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # optimizer
    if args.type =='bls':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta_1,args.beta_2),weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta_1,args.beta_2),weight_decay=args.weight_decay)

    # retrain the model
    if args.resume:
            # resume
        checkpoint = torch.load(args.resume_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        # model.load_state_dict(torch.load(args.resume_dir), strict=False)
    if args.type == 'bls':
        train_bls(args,model,optimizer)
    else:
        train_top(args,model,optimizer)
