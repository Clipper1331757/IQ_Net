# import sqlite3
# import pandas as pd
import os
# import numpy as np
# from Bio import Phylo
# from Bio import AlignIO
# from Bio import SeqIO
# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord
from io import StringIO
# import gc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from tensorflow import keras
# import tensorflow as tf
import itertools
import random

# from sqlalchemy.dialects.mssql.information_schema import columns
# from tqdm import tqdm
# import itertools
# from sklearn.model_selection import train_test_split
"""
Extract pattern frequency from sub-alignments and save it as .csv file
Not used
"""
P4 = list(itertools.permutations([0,1,2,3]))
n_map = {'A':0,'C':1,'G':2,'T':3,'-':4,'N':4}
nu_weights =np.array([125,25,5,1])[:,np.newaxis]
encoding_permutations = np.zeros((24,625),dtype=int)
feature_names = {''.join(p):( np.array([125,25,5,1]) * np.array([n_map[e] for e in p])).sum() for p in itertools.product('ACGT-',repeat=4)}
random.seed(40)
# print(feature_names)

# for i, permu in enumerate(P4):
#     print(permu)
#     permu_indices = np.zeros(625)
#     for s_pattern in feature_names.keys():
#         new_s_pattern = ''.join([s_pattern[i] for i in permu])
#         permu_indices[feature_names[new_s_pattern]]  = feature_names[s_pattern]
#     encoding_permutations[i] = permu_indices
#
# print(encoding_permutations)

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
    # shuffle msa
    items = list(msa.items())
    random.shuffle(items)
    shuffled_msa = dict(items)
    return shuffled_msa

def encode_msa(msa):
    sequences = np.stack([l for l in msa.values()], axis=0)
    encoded = np.zeros((1, 625))
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

def read_tree(tree_file,msa):
    with open(tree_file, 'r') as file:
        tree = file.read().strip()
    # print(tree)
    # print(msa.keys())
    ext_bl = {}
    int_bl = None
    str_left = tree + ''
    int_str_end = None

    while (len(str_left) > 1):
        loc = str_left.find(':')
        candidates = (str_left.find(',', loc), str_left.find(')', loc))
        end = min(candidates) if candidates[0] > 0 and candidates[1] > 0 else max(candidates)
        if str_left[loc - 1] != ')' and loc != 0:
            ext_bl[str_left[:loc].strip('(').strip(',')] = (float(str_left[loc + 1: end]))
        else:
            int_bl = float(str_left[loc + 1: end])
            int_str_end = tree.index(str_left[loc:]) - 1
        str_left = str_left[end + 1:]
    # sorted by taxa name
    # print(ext_bl)
    # ext_bl = sorted(ext_bl.items())
    try:
        s = [i for i in range(len(tree)) if tree[i] == '(' and i < int_str_end][-1]
    except TypeError:
        print('parsing error, returning nan')
        res = np.zeros(5)
        res[:] = np.nan
        return res, np.nan
    bl = []
    for key in msa.keys():
        bl.append(ext_bl[key])
    bl = bl + [int_bl]
    taxa = list(msa.keys())
    top_str = tree[s + 1:int_str_end]
    t1_str, t2_str = top_str.split(',')
    t1 = t1_str.split(':')[0]
    t2 = t2_str.split(':')[0]
    top_taxa = [t1,t2]
    top = -1
    if taxa[0] in top_taxa and taxa[1] in top_taxa:
        top =0
    elif taxa[2] in top_taxa and taxa[3] in top_taxa:
        top =0
    elif taxa[0] in top_taxa and taxa[2] in top_taxa:
        top =1
    elif taxa[1] in top_taxa and taxa[3] in top_taxa:
        top =1
    elif taxa[0] in top_taxa and taxa[3] in top_taxa:
        top =2
    elif taxa[1] in top_taxa and taxa[2] in top_taxa:
        top =2
    # # if one of the two sisters is the first taxa, top are two sisters
    # if ext_bl[0][0] == t1 or ext_bl[0][0] == t2:
    #     top = ''.join(sorted([t1, t2]))
    # else:
    #     # top is the remaining two (one of them being the first taxa)
    #     top = ''.join(sorted([t[0] for t in ext_bl if t[0] not in (t1, t2)]))
    # print(tree)
    # print(top)
    # print(taxa)
    # print('-------------')

    return np.array(bl).reshape(-1), top,taxa


# def get_simu_summary(indices=None, folder_path='data_input/val_trees/', file_extension='nwk', fnames=None):
#     result = []
#     if fnames == None:
#         fnames = [None for _ in indices]
#     else:
#         indices = [None for _ in fnames]
#     for i in tqdm(range(len(indices))):
#         bl, top = read_tree(indices[i], folder_path, file_extension, fnames[i])
#         l = bl.tolist()
#         l.append(top)
#         l.append(indices[i] if indices[i] != None else fnames[i])
#         result.append(l)
#
#     df = pd.DataFrame(result)
#
#     df.columns = ['A', 'B', 'C', 'D', 'int_b', 'top', 'simu_index/name']
#     df.set_index('simu_index/name', inplace=True)
#     return df


tree_path = "./data/reformated_subtrees"

index = pd.read_csv('./index.csv')
nwk_files = []

m,n = index.shape
for i in range(m):
    if index.iloc[i,1] == 0:
        nwk_files.append(index.iloc[i,0])
al_path = './data/subalignments'
count = 0
columns = list(feature_names.keys())
columns.extend( [  'ext_b1','ext_b2', 'ext_b3','ext_b4', 'int_b', 'top','taxon1','taxon2','taxon3','taxon4','tree_id'])
if os.path.exists('./data/data_set.csv'):
    df = pd.read_csv('./data/data_set.csv')
else:
    df = pd.DataFrame(columns=columns)
print(len(nwk_files))
processed_tree = []


start_count = 0

for nwk_file in nwk_files:
    tree_file_path = os.path.join(tree_path, nwk_file)
    file_name = nwk_file.split('.')[0]
    alignments_path = os.path.join(al_path, file_name+'.fasta')
    msa = read_msa(alignments_path)
    encode = list(encode_msa(msa))
    #  
    bl,top,taxa = read_tree(tree_file_path,msa)
    l = encode + list(bl)
    l.append(top)
    l.extend(taxa)
    l.append(file_name)
    if len(file_name)<=1:
        print(tree_file_path)
    sub_df = pd.DataFrame([l],columns = columns)
    # for key in feature_names.keys():
    #     sub_df[key] = [encode[feature_names[key]]]
    # l = [  'A','B', 'C','D', 'int_b']
    # for i in range(5):
    #     sub_df[l[i]] = [bl[i]]
    # sub_df['top'] = [top]
    df= pd.concat([df,sub_df])
    # print(nwk_file)
    # output_path = './data/reformated_subtrees/'+nwk_file
    # modified_newick = move_internal_branch_to_end(file_path)
    # if not modified_newick == None:
    #     write_tree_to_file(modified_newick, output_path)
    count +=1
    processed_tree.append(nwk_file)
    # if count>=100:
    #     break
    if count % 10000 == 0:
        index.loc[index['file_name'].isin(processed_tree), 'processed'] = 1
        # df.to_csv('./data/data_set.csv', index=False)
        csv_path = './data/data_set/data_set_' + str(start_count + count/10000) + '.csv'
        df.to_csv(csv_path, index=False)
        index.to_csv('./index.csv', index=False)
        processed_tree = []
        df = pd.DataFrame(columns=columns)
        print(count)
        # print(df['top'].value_counts())

df.to_csv('./data/data_set/data_set.csv',index = False)
index.loc[index['file_name'].isin(processed_tree), 'processed'] = 1
index.to_csv('./index.csv', index=False)