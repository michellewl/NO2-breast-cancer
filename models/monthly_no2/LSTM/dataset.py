import torch
from torch.utils.data import Dataset
import joblib
import numpy as np

class NO2Dataset(Dataset):
    def __init__(self, sequences_path, targets_path):
        self.inputs = torch.from_numpy(np.load(sequences_path)).float()
        self.targets = torch.from_numpy(np.load(targets_path)).float()

    def __len__(self):
        return self.inputs.size()[0]

    def nfeatures(self):
        return self.inputs.size()[-1]

    def sequence_size(self):
        return self.inputs.size()[1]

    def __getitem__(self, index):
        input = self.inputs[index]
        target = self.targets[index]
        return {"sequences": input, "targets": target}


