# IQ-NET
#### 1. Prerequisites
- Nvidia GPU
#### 2. Description of all python files
- `main.py` - script used to train the model
- `train_top.py` - train the topology classification model, called by `main.py`
- `train_bls.py` - train the branch length prediction model, called by `mian.py`
- `loss_function.py` - customised loss function used for the branch length prediction model
- `dataset.py` - Data Frame Dataset used for training and testing
- `quartet_net.py` - network architecture
- `permutate_pattern_frequence.py` - utility functions called by `quartet_net.py`
- `eval_top` - test the topology classification model
- `eval_bls` - test the branch length prediction model
- `infer_tree.py` - infer four taxa phylogenetic tree in .nwk format from .fasta file
#### 3. First steps
- Download the [Data set](https://drive.google.com/file/d/1NmV3VpgdcaW8SQu3QHMDZR6WzJ4nSHqT/view?usp=sharing)
- Download the [Pre-trained Model](https://drive.google.com/file/d/11yfYPV7zuQKUclLUBxSytUFyQ2VJgqqv/view?usp=sharing)
- Download the [Test alignments](https://drive.google.com/file/d/16W6JBfJzFxfoJjaRXLBHMiraPV-Dx4Pq/view?usp=sharing)
- Put the training data, validation data and testing data in `data/`.
- Put the pre-trained model in `model/`
- Extract the test_align.zip

#### 3. Train the model
- Run `main.py` to train the model, the trained model will be saved in `model` folder.
  - net_name: name of the outputted network model
  - type: bls or top, determine train the branch length regressor or train the topology classifier
  - resume: if resume a pre-trained model and retrain the model, default 0
  - resume_dir: path of the resumed model
  - restore_epoch: restored epochs of the pre-trained model, default 0
  - epochs: training epochs
  - batch_size: batch size
  - lr: learning rate
  - train_dir: path of the training data
  - validation_dir: path of the validation data
  - seed: random seed
  - lr_decay: learning rate decay, after each epoch, lr = lr * lr_decay
  - weight_decay: weight decay of the Adam optimizer
  - alpha: alpha of combined MRE loss, if use combined MRE loss for branch length prediction model

#### 4. Test the model
- Run `eval_top.py` to test the topology prediction model
- Run `eval_bls.py` to test the branch length regressor model.
  - net_dir: path to the saved model
  - test_dir: path of the test data
  - output_dir: path of the output file
- test_dif should be point to a .csv file
- This model will output one CSV file contains predicted branch length or topology.
```bash
python3 eval_top.py --net_name ./model/iq_net_top.pth --test_dir ./data/df_test.csv --output_dir ./df_iq_net_top.csv
```

#### 5. Tree inference
- Run `infer_tree.py` to infer the four-taxa phylogenetic tree.
  - classifier_name: name of the resumed classifier
  - regressor_name: name of the resumed regressor
  - batch_process: if apply batch process; 0 - infer single tree; 1- infer multiple trees
  - alignments_dir: if batch_process = 0, it should be the path of the alignments file, e.g. test_alignment.fasta; if batch_process = 1, it should be path of a folder that store all alignments files that need to be inferred. 
  - output_dir: if batch_process = 0, it should be the path of the tree file, e.g. test_tree.nwk; if batch_process = 1, it should be path of a folder that store all inferred tree files.
  - log_file: name of the log file that store the run time
- Input: .fasta file; Output: .nwk file

#### 6. Other utilities
- Run `train_test_split` to split a data set into two sub-sets.
  - if you want to split dataset to training, validation, and testing data, you could run this script twice
- Run `extract_tree_df.py` to extract tree topology and branch length from tree files and alignments files.
- Run `extract_pattern_frequency_tree_df.py` to extract pattern frequency, tree topology, and branch length from tree files and alignments files.
  - First run `get_tree_index.py` to get all tree file names.
  - Then run `extract_pattern_frequency_tree_df.py` to extract data into multiple .csv files.
  - Run `concat_df.py` to merge multiple .csv files to get the whole dataset.
  - When extract a large number of trees, there may not be enough memory to store the whole df, using these scripts to process the data batch by batch.
- `data_analysis.ipynb` contains codes used for data visualisation and analysis.
  - Should have the processed .csv file.