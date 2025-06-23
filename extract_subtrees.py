import sqlite3
import pandas as pd
import os
import numpy as np
from Bio import Phylo
from Bio import AlignIO
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from io import StringIO

import random
from ete3 import Tree
from Bio import SeqIO
import os
import shutil


seed = 40


def extract_4_taxa_subtrees_ete3(in_tree,num_taxa):
    """
    extract all possible 4-taxon sub-tree
    """
    subtrees= []
    leaves = in_tree.get_leaf_names()
    num_leaves = len(leaves)
    if num_leaves < num_taxa:
        return subtrees

    # compute the number of sub-samples
    # num_subsamples = 1 + int((num_leaves - num_taxa) / 10)
    num_subsamples = max (1, int(num_leaves/num_taxa-1))
    num_subsamples = min(num_subsamples,30)


    # return if there is not enough taxa to sample
    if num_taxa == len(leaves):
        in_tree.unroot()
        subtrees.append(in_tree)
        return subtrees

    subsamples = []
    for _ in range(num_subsamples):
        # Initialize a new subsample
        new_subsample = []

        # Keep sampling until the new subsample is different from all existing subsamples
        while True:
            new_subsample = random.sample(leaves, num_taxa)
            if new_subsample not in subsamples:
                break

        # Add the new_subsample to the list of subsamples
        subsamples.append(new_subsample)

    # extract subtrees
    for i in range(num_subsamples):
        # Find the common ancestor of the taxa in the list
        common_ancestor = in_tree.get_common_ancestor(subsamples[i])

        # Extract the subtree containing the common ancestor and its descendants
        subtree = common_ancestor.copy()

        # Prune the subtree to remove taxa that are not in the list
        subtree.prune(subsamples[i], preserve_branch_length=True)
        # print(subtree.write(format=1))
        # subtree.unroot()
        # print(subtree.write(format=1))
        subtrees.append(subtree)
    return subtrees


def rename_taxa_ete3(subtree, mapping):
    """
    rename species in ete3 subtree
    """
    for leaf in subtree:
        leaf.name = mapping[leaf.name]
    return subtree


def extract_and_rename_alignments(alignments, taxa, mapping):
    """
    extract and rename subalignments
    """
    selected_sequences = []
    for record in alignments:
        if record.id in taxa:
            # rename the taxa to A, B, C, D
            renamed_record = record[:]

            # disable rename
            # renamed_record.id = mapping[record.id]

            renamed_record.description = ''
            selected_sequences.append(renamed_record)
    return selected_sequences


def save_subtree_to_nwk(subtree, output_file):
    """
    save tree to .nwk file
    """
    subtree.write(format=1, outfile=output_file)


def save_alignments_to_fasta(alignments, output_file):
    """
    save alignments as .fasta file
    """
    SeqIO.write(alignments, output_file, "fasta")



def main(nwk_file, fasta_file = "./", output_nwk_dir='./', output_fasta_dir='./',ali_id='',num_taxa = 4):
    # read tree
    tree = Tree(nwk_file)

    # if len(tree.get_leaf_names()) <=4:
    #     return 0

    # extract all 4 taxa subtree
    selected_subtrees = extract_4_taxa_subtrees_ete3(tree,num_taxa)

    # if no valid subtrees
    if len(selected_subtrees) <= 0:
        return

    # if len(subtrees)>10:
    #     print(len(subtrees))

    # selected_subtrees = random.sample(selected_subtrees, k=max(int(0.2 * len(subtrees)),1))

    # read alignments
    alignments = list(SeqIO.parse(fasta_file, "fasta"))

    # rename subtree and alignments, then save to file
    for i, subtree in enumerate(selected_subtrees):
        taxa = [leaf.name for leaf in subtree.get_leaves()]
        random.shuffle(taxa)
        # map species name to A,B,C,D
        mapping = {taxa[j]: chr(65 + j) for j in range(4)}  # chr(65) = 'A'

        # rename subtree
        # renamed_subtree = rename_taxa_ete3(subtree, mapping)
        renamed_subtree = subtree

        # extract and rename alignments
        renamed_alignments = extract_and_rename_alignments(alignments, taxa, mapping)

        # save tree as .nwk file
        output_nwk_file = os.path.join(output_nwk_dir, ali_id+f"_{i + 1}.nwk")
        save_subtree_to_nwk(renamed_subtree, output_nwk_file)

        # save alignments as .fasta file
        output_fasta_file = os.path.join(output_fasta_dir, ali_id+f"_{i + 1}.fasta")
        save_alignments_to_fasta(renamed_alignments, output_fasta_file)
        # return 1

random.seed(seed)

# tree_path = './data/dna_tree_cleaned.csv'
# df_tree = pd.read_csv(tree_path,on_bad_lines='skip',delimiter='\t')
# m,n = df_tree.shape
# for i in range(m):
#     main(nwk_file= df_tree.iloc[i,:]['NEWICK_STRING'], fasta_file = "./", output_nwk_dir='./data/subtrees', output_fasta_dir='',ali_id=df_tree.iloc[i,:]['ALI_ID'])
#     if i >10:
#         break

count = 0
with open('index.txt', 'r') as file:
    while True:
        line = file.readline()
        if not line:
            break
        line = line.strip()
        fasta_path = './data/alignments/' + line +'.fasta'
        nwk_path = './data/tree/' + line +'.nwk'
        output_nwk_dir = './data/subtrees'
        output_fasta_dir = './data/subalignments'
        os.makedirs(output_nwk_dir, exist_ok=True)
        os.makedirs(output_fasta_dir, exist_ok=True)
        main(nwk_path, fasta_path, output_nwk_dir, output_fasta_dir,line)

        #
        # # the code below used for test
        # source_file = nwk_path
        # destination_folder = "./data/test_tree"
        # shutil.copy(source_file, destination_folder)
        #
        # source_file = fasta_path
        #
        # shutil.copy(source_file, destination_folder)
        # count +=1
        # if count >5:
        #     break



