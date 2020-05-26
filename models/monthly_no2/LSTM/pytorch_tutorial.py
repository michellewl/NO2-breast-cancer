import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

torch.manual_seed(1)

lstm = nn.LSTM(3, 3) # Input dimension is 3, output dimension is 3
inputs = [torch.randn(1, 3) for i in range(5)] # make a sequence of length 5, using the pytorch tensor format
# The first axis is the sequence, second indexes the mini-batch, third indexes elements of the input.
# Here, the sequence has length 5, mini-batch dimension has size 1, and there are 3 dimensions of the input.

# initialise the hidden state of the LSTM:
hidden = (torch.randn(1, 1, 3), # sequence length 1, mini-batch dimension size 1, 3 elements of the input
          torch.randn(1, 1, 3))

for i in inputs:
    # Step through the sequence one element at a time
    # After each step, hidden contains the hidden state
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    # print(out)
    # print(hidden)

# Alternatively, we can do the entire sequence at once. The first value returned by LSTM is all of the
# hidden states throughout the sequence. The second value is the most recent hidden state (compare the last
# slice of "out" with "hidden" below - they are the same). This is because
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate by passing it as an argument to
# the LSTM at a later time.
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)

# https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/

# Load the tutorial data set from Seaborn
flight_data = sns.load_dataset("flights")

# Convert the pandas dataframe to a numpy array with float types
all_data_array = flight_data["passengers"].values.astype(float)

# Choose the first 132 records as training data and the last 12 records for testing.
test_data_size = 12
train_data = all_data_array[:-test_data_size]
test_data = all_data_array[-test_data_size:]

# Normalise the data between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalised = scaler.fit_transform(train_data.reshape(-1, 1))
# The tutorial highlights that applying noralisation to the test data risks leaking information
# from the training set to the test set?

# Convert the data set into PyTorch tensors
train_data_normalised = torch.FloatTensor(train_data_normalised).view(-1)  # size = -1

# The final pre-processing step is to convert our training data into sequences and corresponding labels.
# You can use any sequence length and it depends upon the domain knowledge. However, in our dataset it is
# convenient to use a sequence length of 12 since we have monthly data and there are 12 months in a year.
# If we had daily data, a better sequence length would have been 365, i.e. the number of days in a year.
# Therefore, we will set the input sequence length for training to 12.
train_window = 12

# Define a function to take the raw input data and return a list of tuples in which the first element of each
# tuple is the input data for 12 months, and the second element is the output i.e. data for 13th month.
def create_in_out_sequences(input_data, tw):
    in_out_sequence = []
    length = len(input_data)
    for i in range(length-tw):
        train_sequence = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        in_out_sequence.append((train_sequence, train_label))
    return in_out_sequence

# Create sequences and corresponding labels for training
train_in_out_sequences = create_in_out_sequences(train_data_normalised, train_window)

# Create the LSTM model, with LSTM class which inherits from nn.Module class of the PyTorch library.
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        