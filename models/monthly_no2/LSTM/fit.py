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
x_train_norm = x_normaliser.transform(x_train)
y_train_norm = y_normaliser.transform(y_train)

# LSTM model

# Convert dataset to PyTorch tensors
x_train_norm = torch.from_numpy(x_train_norm)
y_train_norm = torch.from_numpy(y_train_norm)
print(f"x train: {x_train_norm.shape}"
      f"\ny train: {y_train_norm.shape}")

# Define function to produce the xy sequences.

def create_xy_sequences(x_array, y_array, tw):
    xy_sequence = []
    for i in range(len(y_array)):
        train_sequence = x_array[i:i+tw].float()
        train_target = y_array[i].float()
        xy_sequence.append((train_sequence, train_target))
    return xy_sequence

training_sequences = create_xy_sequences(x_train_norm, y_train_norm, training_window)
# print(training_sequences[0])

# Create the LSTM model

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=1):
        super().__init__()  # runs init for the Module parent class
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)  # LSTM layers
        self.linear = nn.Linear(hidden_layer_size, output_size)  # Linear layers
        # Hidden cell variable contains previous hidden state and previous cell state.
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        # Pass input sequence through the lstm layer, which outputs the layer output, hidden state and cell state
        # at the current time step.
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        # Pass the lstm output to the linear layer, which calculates the dot product between
        # the layer input and weight matrix.
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# Create model object of the LSTM class, define a loss function, define the optimiser.
model = LSTM(input_size=x_train_norm.shape[1], hidden_layer_size=100, output_size=y_train_norm.shape[1])
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"Model:\n{model}")

# Train the LSTM model
filename = "lstm_model.tar"
save_path = join(load_folder, filename)
torch.manual_seed(1)

num_epochs = 150
training_loss_history = []
# Currently no batches or validation set.

print("Begin training...")
for epoch in range(num_epochs):
    model.train()
    loss_sum = 0  # for storing
    for sequence, target in training_sequences:
        optimiser.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        # Run the forward pass
        y_predict = model(sequence)
        # Compute the loss and gradients
        single_loss = criterion(y_predict, target)
        single_loss.backward()
        # Update the parameters
        optimiser.step()

        loss_sum += single_loss.item()
        training_loss_history.append(loss_sum / len(training_sequences))
        # Print the loss after every 25 epochs
    if epoch % 25 == 0:
        print(f"epoch: {epoch:3} loss: {single_loss.item():10.8f}")
        # Save the model after every 2 epochs
    if epoch % 5 == 0:
        torch.save({
            "total_epochs": epoch,
            "final_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "training_loss_history": training_loss_history
        }, save_path)
print(f"epoch: {epoch:3} loss: {single_loss.item():10.8f}")

