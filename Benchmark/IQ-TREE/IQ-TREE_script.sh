#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=4GB 
#PBS -l jobfs=10GB 
#PBS -q normal 
#PBS -P dx61 
#PBS -l walltime=6:00:00 
#PBS -l storage=scratch/dx61 
#PBS -l wd 

input_dir="./test_align"
output_dir="./test_trees"

mkdir -p "$output_dir"

# support .fas or .fasta
for fasta_file in "$input_dir"/*.fas "$input_dir"/*.fasta; do
    [ -e "$fasta_file" ] || continue

    
    filename=$(basename "$fasta_file")
    filename="${filename%.*}"
    tree_file="${output_dir}/${filename}"
    ./iqtree2 -s  "$fasta_file" -nt AUTO -pre "$tree_file" -seed 1 -quiet 
done