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


# df_sequence = pd.read_csv('./data/dna_sequences.csv')
# df_alignments = pd.read_csv('./data/dna_alignments.csv')
allowed_chars = {'A', 'T', 'C', 'G', '-'}
def replace_disallowed_characters(input_string, allowed_characters):

    result = ''.join([char if char in allowed_characters else '-' for char in input_string])
    return result

sequences_path = './data/dna_sequences.csv'
alignments_path = './data/dna_alignments.csv'
tree_path = './data/dna_tree_cleaned.csv'
df_tree = pd.read_csv(tree_path,on_bad_lines='skip',delimiter='\t')

df_al = pd.read_csv(alignments_path,on_bad_lines='skip',delimiter='\t')


df_seq = pd.read_csv(sequences_path,on_bad_lines='skip',delimiter='\t')
print(df_seq.shape)
print(df_seq.head(5))

print(df_al.columns)
print(df_al.shape)
print(df_al.head(5))

print(df_tree.shape)
print(df_tree.head(5))

m,n = df_tree.shape
valid_files = []
for i in range(m):
    ali_key = df_tree.iloc[i]['ALI_ID']
    if not df_al['ALI_ID'].isin([ali_key]).any():
        continue
    num_taxa = df_al.loc[df_al['ALI_ID'] == ali_key, 'TAXA'].values[0]
    sub_df_seq = df_seq[df_seq['ALI_ID']==ali_key]
    if not sub_df_seq.shape[0] == num_taxa:
        print(df_seq['ALI_ID'])
    # if all taxa can be found
    if sub_df_seq.shape[0] == num_taxa:
        # create SeqRecord
        records = []
        for j in range(sub_df_seq.shape[0]):
            sequence = sub_df_seq.iloc[j]['SEQ']
            sequence = replace_disallowed_characters(sequence, allowed_chars)
            species = sub_df_seq.iloc[j]['SEQ_NAME']
            species = species.replace('/','_')
            species = species.replace('-', '_')
            seq_record = SeqRecord(
                Seq(sequence),
                id=species,
                description=""
            )
            records.append(seq_record)

    #     # save as fasta file
        f_name = './data/alignments/'+ali_key+'.fasta'
        with open(f_name, "w") as fasta_file:
            SeqIO.write(records, fasta_file, "fasta")
        # save tree file
        tree_string = df_tree.loc[df_tree['ALI_ID']==ali_key, 'NEWICK_STRING'].values[0]
        tree_string = tree_string.replace('/','_')
        tree_string = tree_string.replace('-', '_')
        tree = Phylo.read(StringIO(tree_string), "newick")
        f_name = './data/tree/'+ali_key+'.nwk'
        with open(f_name, "w") as nwk_file:
            Phylo.write(tree, nwk_file, "newick")
        valid_files.append(ali_key)
    if i % 1000 ==0:
        print(int(i/1000))


with open('index.txt', 'w') as output:
    for valid_file in valid_files:
        output.write(valid_file + '\n')
