import torch
import torch.nn as nn
import numpy as np
from os.path import join, dirname, realpath
from dataset import NO2Dataset
from torch.utils.data import DataLoader
from lstm_model_class import LSTM

training_window = 3  # consider the last X months of NO2 for each breast cancer diagnosis month

# aggregation = ["min", "max"]
# aggregation = ["mean"]
quantile_step = 0.1  # Make this False if not using.

ccgs = ["NHS Central London (Westminster)", "NHS Richmond"]
ccg = ccgs[0]

# One age category
age_category = "all_ages"
print(f"{ccg}\n{age_category}")

if quantile_step:
    aggregation = f"{int(1/quantile_step)}_quantiles"
load_folder = join(join(join(dirname(realpath(__file__)), ccg), aggregation), f"{training_window}_month_tw")

batch_size = 14
torch.manual_seed(1)

# Load train & test data
training_dataset = NO2Dataset(join(load_folder, "train_val_sequences.npy"), join(load_folder, f"train_val_targets_{age_category}.npy"))
test_dataset = NO2Dataset(join(load_folder, "test_sequences.npy"), join(load_folder, f"test_targets_{age_category}.npy"))

training_dataloader = DataLoader(training_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Load the model
checkpoint = torch.load(join(load_folder,  f"lstm_model_{age_category}.tar"))
model = LSTM(input_size=training_dataset.nfeatures(), hidden_layer_size=100)
model.load_state_dict(checkpoint["best_state_dict"])
print(model)
model.eval()