import pandas as pd
import os
import argparse

# get the file name of all tree files, convert it to a .csv file
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_folder', type=str, default='./tree',help = 'path to the folder contains tree files', required=False)

    parser.add_argument('--output_dir', type=str, default='./index.csv',help = 'path to the output .csv file', required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tree_path = args.tree_folder
    nwk_files = [f for f in os.listdir(tree_path) if f.endswith('.nwk')]
    index = pd.DataFrame(columns=['file_name', 'processed'])
    index['file_name'] = nwk_files
    index['processed'] = [0 for _ in range(len(nwk_files))]
    index.to_csv(args.output_dir, index=False)
