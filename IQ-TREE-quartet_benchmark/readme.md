# IQ-TREE Quartet Tree Inference

This script performs quartet tree phylogenetic tree inference using **IQ-TREE 2** for all FASTA alignments in the `alignments/` directory.

Each alignment is processed sequentially. The inferred trees and associated output files are written to the `test_trees/` directory. After all analyses are complete, the output directory is compressed into a `tar.gz` archive.

Before running the script, please download **IQ-TREE 2 (Legacy Release v2.4.0)** from the official website:

https://iqtree.github.io/



Place the executable (`iqtree2`) in the working directory (or modify the script to point to its location).
---

## Directory Structure

Before running the script, the working directory should have the following structure:

```
project/
│
├── iqtree2
├── run_iqtree.pbs
├── alignments/
│   ├── alignment1.fasta
│   ├── alignment2.fasta
│   └── ...
└── test_trees/
```

If the `test_trees` directory does not exist, create it first:

```bash
mkdir test_trees
```

---

# Running on Linux

The PBS directives at the beginning of the script can simply be ignored if running outside a PBS scheduler.

Create a simplified shell script:

```bash
#!/bin/bash

mkdir -p test_trees

for aln in ./alignments/*.fasta; do
    aln_name=$(basename "$aln" .fasta)

    ./iqtree2 \
        -s "$aln" \
        -nt 1 \
        -pre ./test_trees/"$aln_name" \
        -seed 1 \
        -quiet
done
```

---

# IQ-TREE Options

| Option | Description |
|---------|-------------|
| `-s` | Input alignment |
| `-nt 1` | Use one CPU thread |
| `-pre` | Output file prefix |
| `-seed 1` | Fixed random seed for reproducibility |
| `-quiet` | Suppress verbose console output |

---

# Requirements

- Linux
- IQ-TREE 2 executable (`iqtree2`)
- FASTA alignments stored in

```
alignments/
```

---

# Notes

- Results are written into `test_trees/`.
- After successful completion, the results are compressed into `test_trees.tar.gz`.
- The script records the total execution time in `iq_tree_runtime.log`.