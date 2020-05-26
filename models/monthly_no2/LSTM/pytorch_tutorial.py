# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

# Alternatively, we can do the entire sequence at once.
# The first value returned by LSTM is all of the hidden states throughout the
# sequence. The second value is the most recent hidden state (compare the last
# slice of "out" with "hidden" below - they are the same).
# This is because
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate by
# passing it as an argument to the LSTM at a later time.
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)