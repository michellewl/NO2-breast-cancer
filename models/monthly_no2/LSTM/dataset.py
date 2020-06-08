import torch
from torch.utils.data import Dataset
import joblib
import numpy as np

class NO2Dataset(Dataset):
    def __init__(self, sequences_path, targets_path, noise_std=False):
        self.inputs = torch.from_numpy(np.load(sequences_path)).float()
        self.targets = torch.from_numpy(np.load(targets_path)).float()
        self.noise_std = noise_std  # Standard deviation of Gaussian noise

    def __len__(self):
        return self.inputs.size()[0]

    def nfeatures(self):
        return self.inputs.size()[-1]

    def sequence_size(self):
        return self.inputs.size()[1]

    def __getitem__(self, index):
        input = self.inputs[index]
        target = self.targets[index]
        if self.noise_std:
            noise = torch.randn_like(input)*self.noise_std
            input = input + noise
        return {"sequences": input, "targets": target}


