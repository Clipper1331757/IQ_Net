#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l jobfs=10GB
#PBS -q normal
#PBS -P dx61
#PBS -l walltime=6:00:00
#PBS -l storage=scratch/dx61
#PBS -l wd

echo "Job started at $(date)"
start=$(date +%s)

for aln in ./alignments/*.fasta; do
  aln_name=$(basename "$aln" .fasta)
  ./iqtree2 -s "$aln" -nt 1 -pre ./test_trees/"$aln_name" -seed 1 -quiet
done

end=$(date +%s)
runtime=$((end - start))
echo "Job finished at $(date)"
echo "Script runtime: $runtime seconds" > iq_tree_runtime.log
echo "Runtime saved to log file."
echo "Compressing ./test_trees ..."
tar -czvf test_trees.tar.gz test_trees
if [ $? -eq 0 ]; then
  rm -rf ./test_trees
else
  echo "Compression failed. Keeping original folder."
fi