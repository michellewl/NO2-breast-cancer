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
# print(out)
# print(hidden)

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
    # Though the sequence length is 12, for each month we have 1 dimensional value (# of passengers) so input size is 1.
    # We specify one hidden layer with 100 neurons.
    # Output size is 1 because we predict 1-dimensional value
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        # Create LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        # Create linear layers
        self.linear = nn.Linear(hidden_layer_size, output_size)
        # Create hidden cell variable containing previous hidden state and previous cell state.
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))
    def forward(self, input_seq):
        # Pass input sequence through the lstm layer, which outputs the layer output, hidden state and cell state
        # at the current time step.
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        # Pass the lstm output to the linear layer, which calculates the dot product between
        # the layer input and weight matrix - predicted number of passengers, in this case.
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# Create model object of the LSTM class, define a loss function, define the optimiser.
model = LSTM()
loss_function = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"Model:\n{model}")

# Train the model for 150 epochs
filename = "tutorial_model.tar"
epochs = 150
training_loss_history =[]

for epoch in range(epochs):
    model.train()
    loss_sum = 0 # for storing
    for sequence, labels in train_in_out_sequences:
        # Prevent PyTorch from accumulating gradients.
        optimiser.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        # Run the forward pass
        y_predict = model(sequence)
        # Compute the loss and gradients
        single_loss = loss_function(y_predict, labels)
        single_loss.backward()
        # Update the parameters
        optimiser.step()

        loss_sum += single_loss.item()
        training_loss_history.append(loss_sum/len(train_in_out_sequences))
    # Print the loss after every 25 epochs
    if epoch % 25 == 1:
        print(f"epoch: {epoch:3} loss: {single_loss.item():10.8f}")
    # Save the model after every 2 epochs
    if epoch % 2 == 0:
        torch.save({
            "total_epochs": epoch,
            "final_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "training_loss_history": training_loss_history
        }, filename)
print(f"epoch: {epoch:3} loss: {single_loss.item():10.8f}")

# Re-load the saved model as a check point
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint["final_state_dict"])
print("model reloaded")

# Making predictions
# To make predictions on the test set the model uses sequence length 12,
# so we filter the last 12 values  from the training set.

future_predict = 12  # number of elements in the test set
test_inputs = train_data_normalised[-train_window:].tolist()

# The initial 12 items in test_inputs will be used to predict the first test item. The predicted value is then appended
# to the test_inputs. This is repeated for each test prediction, taking the last 12 items in test_inputs each time.

model.eval()

for i in range(future_predict):
    sequence = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(sequence).item())

test_predicted = test_inputs[future_predict:]
# De-normalise the predicted values
test_predicted = scaler.inverse_transform(np.array(test_predicted).reshape(-1, 1))

# Plot the prediction
plot_x = np.arange(132, 144, 1)
fig, ax = plt.subplots()
ax.plot(flight_data["passengers"])
ax.plot(plot_x, test_predicted)
fig.suptitle("Month vs Passengers")
ax.set_ylabel("Total Passengers")
plt.show()
