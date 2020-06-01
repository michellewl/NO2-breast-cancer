import torch
import numpy as np
import pandas as pd
from os.path import join, dirname, realpath
from dataset import NO2Dataset
from torch.utils.data import DataLoader
from lstm_model_class import LSTM
import joblib
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

training_window = 3  # consider the last X months of NO2 for each breast cancer diagnosis month

# aggregation = ["min", "max"]
# aggregation = ["mean"]
quantile_step = 0.1  # Make this False if not using.

ccgs = ["NHS Central London (Westminster)", "NHS Richmond"]
ccg = ccgs[0]
test_year = 2017

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

training_losses = checkpoint["training_loss_history"]
val_losses = checkpoint["validation_loss_history"]
best_epoch = checkpoint["best_epoch"]
epochs = checkpoint["total_epochs"]

# Plot training and validation loss history and annotate the epoch with best validation loss

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(range(epochs+1), training_losses, label="training loss")
ax.plot(range(epochs+1), val_losses, label="validation loss")
ax.scatter(best_epoch, min(val_losses))
plt.annotate(f"epoch {best_epoch}", (best_epoch*1.05, min(val_losses)))
plt.legend()
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.title(f"Training loss")
# plt.show()
fig.tight_layout()
fig.savefig(join(load_folder, f"loss_history_{age_category}.png"), dpi=fig.dpi)