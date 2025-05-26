import os
import re
import pandas as pd
import numpy as np
import argparse


"""
Extract branch length and topology from predicted trees and save it as .csv file
"""
# get the list of taxa
def get_taxa(msa_file):
    with open(msa_file, 'r') as f:
        lines = f.readlines()
    taxa = []
    name = None
    for s in lines:
        if s[0] == '>':
            name = s[1:].strip()
            taxa.append(name)
    return taxa

def get_substring_between_chars(input_str, start_char, end_char):

    start_index = input_str.find(start_char)
    end_index = input_str.find(end_char, start_index + 1)

    if start_index != -1 and end_index != -1:
        return input_str[start_index + 1:end_index]
    else:
        return None


def extract_content_in_parentheses(input_str):

    match = re.search(r'\((.*?)\)', input_str)
    if match:
        return match.group(1)  # return matched content
    else:
        return None

def read_nwk_as_string(file_path):
    with open(file_path, 'r') as file:
        newick_string = file.read().strip()  # read file and remove the space
    return newick_string

def get_tree_dict(nwk_file,char_list):

    tree_dict = {}

    nwk_string = read_nwk_as_string(nwk_file)
    # print(nwk_string)
    nwk_string = nwk_string[1:]
    nwk_string = nwk_string[:-2]
    # print(nwk_string)
    # get external branch
    for c in char_list:
        sub_string = get_substring_between_chars(nwk_string,c,',')
        if sub_string == None:
            sub_string = get_substring_between_chars(nwk_string,c,')')
        if sub_string == None:
            start_index = nwk_string.find(c)
            # print(start_index)
            sub_string = nwk_string[start_index + 1:]

        # print(sub_string)
        sub_string = sub_string[1:]
        if sub_string =='':
           sub_string
        if ')' in sub_string:
            start_index = sub_string.find(')')
            sub_string = sub_string[:start_index]
        tree_dict[c] = sub_string
    # get internal branch
    sub_string = get_substring_between_chars(nwk_string, ')', ',')
    if sub_string == None:
        start_index = nwk_string.find(')')
        # print(start_index)
        sub_string = nwk_string[start_index+1:]
    sub_string = sub_string[1:]
    tree_dict['int_b'] = sub_string
    # print(tree_dict)

    internal_string = extract_content_in_parentheses(nwk_string)
    # print(internal_string)
    pair1 = set()
    for c in internal_string:
        if c in char_list:
            pair1.add(c)

    pair2 = set(char_list)
    pair2 = pair2 - pair1
    tree_dict['sisters'] = [pair1,pair2]
    return tree_dict



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_folder', type=str, default='./trees',help = 'path to the folder contains tree files', required=False)
    parser.add_argument('--alignments_folder', type=str, default='./alignments',help = 'path to the folder contains alignments', required=False)
    parser.add_argument('--tree_file_extension', type=str, default='.treefile',help='tree file extensions, .treefile or .nwk', required=False)

    parser.add_argument('--output_dir', type=str, default='./val_data.csv', help='path to the output .csv file', required=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    alignments_path = args.tree_folder
    tree_path = args.alignments_folder

    columns = ['ext_b1', 'ext_b2', 'ext_b3', 'ext_b4', 'int_b', 'top', 'taxon1', 'taxon2', 'taxon3', 'taxon4',
               'tree_id']
    df = pd.DataFrame(columns=columns)

    # read all tree files
    folder_path = tree_path
    tree_files = [f for f in os.listdir(folder_path) if f.endswith(args.tree_file_extension)]
    for tree_file in tree_files:
        tree_file_path = os.path.join(folder_path, tree_file)
        alignments_file = tree_file.split('.')[0]
        alignments_file = alignments_file + '.fasta'
        alignments_file_path = os.path.join(alignments_path, alignments_file)
        taxa = get_taxa(alignments_file_path)
        tree_id = tree_file.split('.')[0]
        tree_dict = get_tree_dict(tree_file_path, taxa)
        ls = []
        # add internal branch length
        for i in range(4):
            ls.append(tree_dict[taxa[i]])
        ls.append(tree_dict['int_b'])
        taxa_pairs = [set([taxa[0], taxa[1]]), set([taxa[0], taxa[2]]), set([taxa[0], taxa[3]])]
        # get topology
        top = -1
        for i in range(3):
            sister = taxa_pairs[i]
            if sister.issubset(tree_dict['sisters'][0]):
                top = i
                break
            if sister.issubset(tree_dict['sisters'][1]):
                top = i
                break
        ls.append(top)
        ls.extend(taxa)
        ls.append(tree_id)
        sub_df = pd.DataFrame([ls], columns=columns)
        if df.shape[0] <= 0:
            df = sub_df
        else:
            df = pd.concat([df, sub_df])

    df.to_csv(args.output_dir, index=False)

