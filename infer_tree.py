import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from iq_net import IQ_Net_bls,IQ_Net_top
import argparse
import os
from datetime import datetime
import numpy as np
import itertools
import random



P4 = list(itertools.permutations([0,1,2,3]))
n_map = {'A':0,'C':1,'G':2,'T':3,'-':4,'N':4}
nu_weights =np.array([125,25,5,1])[:,np.newaxis]
encoding_permutations = np.zeros((24,625),dtype=int)
feature_names = {''.join(p):( np.array([125,25,5,1]) * np.array([n_map[e] for e in p])).sum() for p in itertools.product('ACGT-',repeat=4)}

# random.seed(40)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def read_msa(msa_file, fasta = True):
    with open(msa_file, 'r') as f:
        lines = f.readlines()
    msa = {}
    if not fasta:
        lines = lines[1:]
        for s in lines:
            name,seq = s.split()
            msa[name.strip()] = np.array([n_map[c] for c in seq.strip()])
    else:
        name = None
        ls = []
        for s in lines:
            if s[0] == '>':
                if len(ls) > 0:
                    msa[name] = np.array(ls)
                ls = []
                name = s[1:].strip()
            else:
                for c in s.strip():
                    ls.append(n_map[c])
        msa[name] = np.array(ls)

    return msa

def encode_msa(msa):
    sequences = np.stack([l for l in msa.values()], axis=0)
    # encoded = np.zeros((1, 625))
    # get original encoding
    patter_nums = (sequences * nu_weights).sum(axis=0)
    pattern_counts = np.zeros(625, dtype=int)
    uniques, counts = np.unique(patter_nums, return_counts=True)
    for i in range(len(uniques)):
        pattern_counts[uniques[i]] = counts[i]
    # get all permutations
    # for row in range(24):
    #     encoded[row] = pattern_counts[encoding_permutations[row]]
    encoded = pattern_counts / sequences.shape[1]

    # for permu_i,permu in enumerate(P4):
    #     new_msa = sequences[list(permu)]
    #     patter_nums = (new_msa * nu_weights).sum(axis=0)
    #     pattern_counts = np.zeros(625, dtype=int)
    #     uniques, counts = np.unique(patter_nums,return_counts=True)
    #     for i in range(len(uniques)):
    #         pattern_counts[uniques[i]] = counts[i]
    #     encoded[permu_i] = pattern_counts
    # encoded = encoded / sequences.shape[1]
    return encoded.reshape(-1)

def infer_tree(msa_file, output_path, top_classifier, bls_predictor,fasta = True,device = 'gpu'):
    msa = read_msa(msa_file,  fasta = fasta)
    taxa = list(msa.keys())
    pattern_frequency = encode_msa(msa)
    pattern_frequency = torch.tensor(pattern_frequency).to(device).float().unsqueeze(0)
    top = top_classifier(pattern_frequency)
    bls = bls_predictor(pattern_frequency)
    # print(bls)
    top = top.cpu().detach().numpy().reshape(-1)
    bls = bls.cpu().detach().numpy().reshape(-1)
    # print(bls)
    # print(top)
    tree_map = {}
    for i in range(4):
        tree_map[taxa[i]] = bls[i]
    tree_map['int'] = bls[-1]
    top = np.argmax(top)
    # print(top)
    l1, l2, l3, l4, l5 = bls
    n1, n2, n3, n4 = taxa
    if top == 0:
        nwk = '({}:{},{}:{},({}:{},{}:{}):{});'.format(n1, l1, n2, l2, n3, l3, n4, l4, l5)
    elif top == 1:
        nwk = '({}:{},{}:{},({}:{},{}:{}):{});'.format(n1, l1, n3, l3, n2, l2, n4, l4, l5)
    else:
        nwk = '({}:{},{}:{},({}:{},{}:{}):{});'.format(n1, l1, n4, l4, n2, l2, n3, l3, l5)
    if output_path != None:
        with open(output_path,'w') as f:
            f.write(nwk)
    return nwk

def safe_torch_load(path, device):
    """
    map torch to CPU if GPU is not available
    """
    kw = {}
    if device.type == "cpu":
        kw["map_location"] = torch.device("cpu")

    try:
        return torch.load(path, weights_only=True, **kw)
    except TypeError:
        return torch.load(path, **kw)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier_name', type=str, default='iq_net_top',help = 'name of the classifier network', required=False)
    parser.add_argument('--regressor_name', type=str, default='iq_net_bls',help = 'name of the regressor network', required=False)

    parser.add_argument('--batch_process', type=int, default=0,help='0 for infer single tree, 1 for infer multiple trees', required=False)
    parser.add_argument('--alignments_dir', type=str,default='./test_alignmens/Bergsten_2013-dna_16S_1.fasta',help='path to the alignments file or folder', required=False)
    parser.add_argument('--output_dir', type=str,default='./iq_net_test_trees/Bergsten_2013-dna_16S_1_iq_tree.nwk', help='path of the output treefile or folder', required=False)
    parser.add_argument('--log_file', type=str, default='./run_time.log', help='path of the log file',required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    start = datetime.now()
    # load classifier
    top_classifier = IQ_Net_top().to(device)
    top_classifier = torch.nn.DataParallel(top_classifier)
    resume_dir = './model/'+args.classifier_name+'.pth'
    # checkpoint = torch.load(resume_dir, weights_only=True)
    checkpoint = safe_torch_load(resume_dir, device)
    top_classifier.load_state_dict(checkpoint['model_state_dict'])

    # load regressor
    bls_predictor = IQ_Net_bls().to(device)
    bls_predictor = torch.nn.DataParallel(bls_predictor)
    resume_dir = './model/'+args.regressor_name+'.pth'
    # checkpoint = torch.load(resume_dir, weights_only=True)
    checkpoint = safe_torch_load(resume_dir, device)
    bls_predictor.load_state_dict(checkpoint['model_state_dict'])
    # set the model to evaluation mode
    top_classifier.eval()
    bls_predictor.eval()
    # batch process
    if args.batch_process == 1:
        folder_path = args.alignments_dir
        # find all fasta files under a certain folder
        fasta_files = [f for f in os.listdir(folder_path) if f.endswith(".fasta")]
        # infer all trees
        # count = 0
        for tree_id in fasta_files:
            msa_file = args.alignments_dir +'/' + tree_id
            output_path = args.output_dir + '/' + tree_id.split('.')[0] +'.nwk'
            nwk = infer_tree(msa_file,output_path,top_classifier,bls_predictor,device = device)
            # count +=1
            # if count >=10:
            #     break
    # infer single tree
    if args.batch_process == 0:
        nwk = infer_tree(args.alignments_dir,args.output_dir,top_classifier,bls_predictor,device = device)

    end = datetime.now()
    time_difference = end - start
    with open(args.log_file, "a") as log_file:
        log_file.write(f"Run time: {time_difference} \n")


