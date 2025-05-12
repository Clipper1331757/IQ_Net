import argparse
import torch
from quartet_net import Quartet_Net_bls,Quartet_Net_top
from train_top import train_top
import numpy as np
from train_bls import train_bls

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', type=str, default='iq_net_top',help = 'name of the network', required=False)
    parser.add_argument('--type', type=str, default='bls', help='type of the network, bls or top', required=False)
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model',required=False)
    parser.add_argument('--resume_dir', type=str, default='./model/iq_net_top.pth', help='dir of resumed model', required=False)
    parser.add_argument('--restore_epoch', type=int, default=0, help='restore epochs',required=False)

    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs',required=False)
    parser.add_argument('--batch_size', type=int, default=64,required=False)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate',required=False)


    parser.add_argument('--train_dir',type=str,default='./data/data_train_v3.csv',required=False)
    parser.add_argument('--validation_dir',type=str,default='./data/data_val_v3.csv',required=False)

    parser.add_argument('--seed', type=int, default=757, required=False)
    parser.add_argument('--lr_decay', type=float, default=0.95, required=False)
    parser.add_argument('--weight_decay', type=float, default=1e-6, required=False)
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
        model = Quartet_Net_bls()
    else:
        model = Quartet_Net_top()

    model = model.to(device)
    model = torch.nn.DataParallel(model)


    # optimizer
    if args.type =='bls':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.95,0.9),weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.95,0.99))

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
