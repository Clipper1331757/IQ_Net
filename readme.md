# IQ-NET
#### 1. Prerequisites
- Nvidia GPU

#### 2. Description of all python files
- `main.py` - script used to train the model
- `train_top.py` - train the topology classification model, called by `main.py`
- `train_bls.py` - train the branch length prediction model, called by `main.py`
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

#### 4. Train the model
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

#### 5. Test the model
- Run `eval_top.py` to test the topology prediction model

########## BEGIN OF NHAN's comments #############


We should add an example command how to run the test as in the following.

	    python eval_top.py --net_name "iq_net_top" --test_dir "./data/data_test_v3.csv" --output_dir "./iq_net_top_df.csv"
	    
where 

  - `--net_name "iq_net_top"` specifies a preptrained network stored at `./model/iq_net_top.pth`
  - `--test_dir "./data/data_test_v3.csv"` specifies the path to the testing data in CSV format.
  - `--output_dir "./iq_net_top_df.csv"` specifies the path to the output CSV file.
	    
	**Other comments:**
	- instead of specifying a net\_name and hard code the path folder that contains the pretrained models as `./model/<net_name>.pth`, we could allow users to specify `--net <file_path>`, which is more flexible and convenient for users.
	- `--test_dir` is misleading - we ask users to specify a file (not a directory), `--test_file` or `--input_file` could be better.
	- `--output_dir` is misleading - we ask users to specify a file (not a directory), `--output` or `--output_file` could be better.
	- The input file here is actually a CSV file while the original testing data should be a set of alignments and a set of the corresponding trees. So, I think you need to add a script (and an example command) to generate a CSV file from the original testing data.
	- The output is a CSV file that contains multiple columns but I don't know exactly how to interpret it. It would be great if you explain the output file and how we can know whether IQ-NET returns a correct or wrong tree for each example.
	- Assuming that you're writing an intruction to show the users how to test your networks:
	  + Input: a folder that contains a set of alignments and a folder that contains a set of the corresponding trees.
	  + Output: Which samples are inferred accurately or wrongly. The overall accuracy X%.
	  + The intruction should includes sample command to run scripts and explanation of the input parameters.

########## END OF NHAN's comments #############

- Run `eval_bls.py` to test the branch length regressor model.
  - net_name: name of the resumed network
  - test_dir: path of the test data
  - output_dir: path of the output file
- This model will output one CSV file contains predicted branch length or topology.

########## BEGIN OF NHAN's comments #############

Same comments as mentioned above.

Beside, I have corrected the wrong default values for `net_name` and `output_dir` in `eval_bls.py`.


########## END OF NHAN's comments #############



#### 6. Tree inference
- Run `infer_tree.py` to infer the four-taxa phylogenetic tree.
  - classifier_name: name of the resumed classifier
  - regressor_name: name of the resumed regressor
  - batch_process: if apply batch process; 0 - infer single tree; 1- infer multiple trees
  - alignments_dir: if batch_process = 0, it should be the path of the alignments file, e.g. test_alignment.fasta; if batch_process = 1, it should be path of a folder that store all alignments files that need to be inferred. 
  - output_dir: if batch_process = 0, it should be the path of the tree file, e.g. test_tree.nwk; if batch_process = 1, it should be path of a folder that store all inferred tree files.
  - log_file: name of the log file that store the run time

  
########## BEGIN OF NHAN's comments #############

I guess this section is for end-users, who want to input (a) alignment(s) and want to infer (a) tree(s) with branch lengths. For testing (to obtain the accuracy and to know exactly which inferred tree is correct or wrong), we need to use the two sections above.

########## END OF NHAN's comments #############


########## BEGIN OF NHAN's comments #############

As we also ask them to benchmark other methods on their data, you should also create relevant material (e.g., scripts, example commands, and step-by-step instruction) to conduct the testing on these methods given a folder that contains a set of alignments and a folder that contains a set of the corresponding trees.

########## END OF NHAN's comments #############