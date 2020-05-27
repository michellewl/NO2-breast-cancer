from os.path import join, dirname, realpath
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import numpy as np
import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import join, dirname, realpath, exists
import re
from sklearn.preprocessing import StandardScaler
import joblib
import datetime as dt
from dateutil.relativedelta import relativedelta

training_window = 3  # consider the last X months of NO2 for each breast cancer diagnosis month

# aggregation = ["min", "max"]
# aggregation = ["mean"]
quantile_step = 0.1  # Make this False if not using.

ccgs = ["NHS Central London (Westminster)", "NHS Richmond"]
ccg = ccgs[0]
test_year = 2017

if quantile_step:
    aggregation = [f"{int(method*100)}_quantile" for method in np.round(np.arange(0, 1+quantile_step, quantile_step), 2).tolist()]
print(aggregation)

# One age category
age_category = "all_ages"
print(f"{ccg}\n{age_category}")

# Load the arrays
if quantile_step:
    aggregation = [str(len(aggregation)-1), "quantiles"]
load_folder = join(join(join(dirname(realpath(__file__)), ccg), "_".join(aggregation)), f"{training_window}_month_tw")
x_train, x_test = np.load(join(load_folder, "x_train.npy")), np.load(join(load_folder, "x_test.npy"))
y_train, y_test = np.load(join(load_folder, f"y_{age_category}_train.npy")), np.load(join(load_folder, f"y_{age_category}_test.npy"))

# Load normalisation
x_normaliser, y_normaliser = joblib.load(join(load_folder, "x_normaliser.sav")), \
                             joblib.load(join(load_folder, f"y_{age_category}_normaliser.sav"))


print(f"x train: {x_train.shape}"
      f"\ny train: {y_train.shape}"
      f"\nx test: {x_test.shape}"
      f"\ny test: {y_test.shape}")

# Normalise input and output training data
x_train = x_normaliser.transform(x_train)
y_train = y_normaliser.transform(y_train)

exit()
# LSTM model

# Convert dataset to PyTorch tensors



# Define function to produce the xy sequences.

def create_xy_sequences(x_array, y_array, tw):
    xy_sequence = []
    for i in range(len(x_array)):
        train_sequence = x_array[i:i+tw]
        train_target = y_array[i]
        xy_sequence.append((train_sequence, train_target))
    return xy_sequence
