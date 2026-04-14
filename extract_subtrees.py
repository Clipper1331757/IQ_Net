import pandas as pd
import argparse
import random
from ete3 import Tree
from Bio import SeqIO
import os


def extract_4_taxa_subtrees_ete3(in_tree,num_taxa=4,max_sample = 30):
    """
    extract all possible 4-taxon sub-tree
    """
    subtrees= []
    leaves = in_tree.get_leaf_names()
    num_leaves = len(leaves)
    if num_leaves < num_taxa:
        return subtrees

    # compute the number of sub-samples
    num_subsamples = max (1, int(num_leaves/num_taxa-1))
    num_subsamples = min(num_subsamples,max_sample)

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


def extract_and_rename_alignments(alignments, taxa,mapping):
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



def extract(nwk_file, fasta_file = "./", output_nwk_dir='./', output_fasta_dir='./',ali_id='',max_sample = 30,num_taxa = 4):
    # read tree
    tree = Tree(nwk_file)

    # if len(tree.get_leaf_names()) <=4:
    #     return 0

    # extract all 4 taxa subtree
    selected_subtrees = extract_4_taxa_subtrees_ete3(tree,num_taxa = num_taxa,max_sample = max_sample)

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_folder', type=str, default='./treebase/trees',help = 'path to the folder contains tree files', required=False)
    parser.add_argument('--alignments_folder', type=str, default='./treebase/alignments',help = 'path to the folder contains alignments', required=False)
    parser.add_argument('--output_tree_folder', type=str, default='./treebase/subtrees', required=False)
    parser.add_argument('--output_alignments_folder', type=str, default='./treebase/subalignments', required=False)
    parser.add_argument('--max_sample', type=int, default=30, required=False)
    parser.add_argument('--num_taxa', type=int, default=4, required=False)
    parser.add_argument('--seed', type=int, default=40, help='ramdom seed',required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    tree_folder = args.tree_folder
    alignments_folder = args.alignments_folder

    if not os.path.exists(args.output_tree_folder):
        os.makedirs(args.output_tree_folder)

    if not os.path.exists(args.output_alignments_folder):
        os.makedirs(args.output_alignments_folder)

    files = os.listdir(tree_folder)
    for file in files:
        tree_id = file.split('.')[0]
        alignments_file = os.path.join(alignments_folder, tree_id+'.fasta')

        if not os.path.exists(alignments_file):
            continue
        tree_file = os.path.join(tree_folder, file)
        extract(tree_file, alignments_file, output_nwk_dir= args.output_tree_folder, output_fasta_dir= args.output_alignments_folder,ali_id = tree_id ,max_sample= args.max_sample,num_taxa = args.num_taxa)




