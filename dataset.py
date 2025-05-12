
import torch
from torch.utils.data import Dataset



# create DataFrameDataset
class PatternFrequencyDataset_top(Dataset):

    def __init__(self,data):
        self.data = torch.tensor(data.iloc[:,:].values)


    # a function to get items by index, return site pattern frequency and tree topology
    def __getitem__(self, index):
        features = self.data[index,:-1]
        label = self.data[index, -1]
        return features, label

    # a function to count samples
    def __len__(self):
        return self.data.shape[0]


class PatternFrequencyDataset_bls(Dataset):

    def __init__(self,data):
        self.data = torch.tensor(data.iloc[:,:].values)


    # a function to get items by index, return site pattern frequency and branch length
    def __getitem__(self, index):
        features = self.data[index,:-5]
        label = self.data[index, -5:]
        return features, label

    # a function to count samples
    def __len__(self):
        return self.data.shape[0]
