import os
import numpy as np
import pandas as pd
import itertools
import argparse

"""
merge multiple .csv files
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='./data_sets',help = 'path to the folder contains tree files', required=False)
    parser.add_argument('--output_dir', type=str, default='./data_set.csv', help='path to the output .csv file', required=False)


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    P4 = list(itertools.permutations([0, 1, 2, 3]))
    n_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4, 'N': 4}
    nu_weights = np.array([125, 25, 5, 1])[:, np.newaxis]
    encoding_permutations = np.zeros((24, 625), dtype=int)
    feature_names = {''.join(p): (np.array([125, 25, 5, 1]) * np.array([n_map[e] for e in p])).sum() for p in
                     itertools.product('ACGT-', repeat=4)}

    columns = list(feature_names.keys())
    columns.extend(
        ['ext_b1', 'ext_b2', 'ext_b3', 'ext_b4', 'int_b', 'top', 'taxon1', 'taxon2', 'taxon3', 'taxon4', 'tree_id'])

    df = pd.DataFrame(columns=columns)

    folder_path = args.dataset_folder

    # get all sub df
    df_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for df_file in df_files:
        file_path = os.path.join(folder_path, df_file)
        sub_df = pd.read_csv(file_path)
        df = pd.concat([df, sub_df])

    df.to_csv(args.output_dir, index=False)

    