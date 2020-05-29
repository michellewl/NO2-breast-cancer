import torch
import torch.nn as nn
import numpy as np
from os.path import join, dirname, realpath
from sklearn.model_selection import train_test_split
import joblib
from copy import deepcopy
from dataset import NO2Dataset
from torch.utils.data import DataLoader

training_window = 3  # consider the last X months of NO2 for each breast cancer diagnosis month

# aggregation = ["min", "max"]
# aggregation = ["mean"]
quantile_step = 0.1  # Make this False if not using.

ccgs = ["NHS Central London (Westminster)", "NHS Richmond"]
ccg = ccgs[1]
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


# print(f"Data loaded:"
#       f"\nx train {x_train.shape}"
#       f"\ny train {y_train.shape}"
#       f"\nx test {x_test.shape}"
#       f"\ny test {y_test.shape}")

# Normalise input and output training data
x_train_norm = x_normaliser.transform(x_train)
y_train_norm = y_normaliser.transform(y_train).squeeze()

# LSTM model

# Convert dataset to PyTorch tensors
# x_train_norm = torch.from_numpy(x_train_norm)
# y_train_norm = torch.from_numpy(y_train_norm)
# print(f"x train: {x_train_norm.shape}"
#       f"\ny train: {y_train_norm.shape}")

# Define function to produce the xy sequences.

def create_x_sequences(x_array, num_sequences, tw):
    input_sequences = []
    for i in range(num_sequences):
        train_sequence = x_array[i:i+tw]
        input_sequences.append(train_sequence)
    input_sequences = np.stack(input_sequences, axis=0)
    return input_sequences

train_val_inputs = create_x_sequences(x_train_norm, len(y_train_norm), training_window)
# print(train_val_inputs.shape, y_train_norm.shape)
validation_size = 0.2
training_sequences, validation_sequences, training_targets, validation_targets = train_test_split(train_val_inputs, y_train_norm, test_size=validation_size, random_state=1)
print(f"Training sequences {training_sequences.shape}\nValidation sequences {validation_sequences.shape}")
print(f"Training targets {training_targets.shape}\nValidation targets {validation_targets.shape}")

save_folder = load_folder
np.save(join(save_folder, "training_sequences.npy"), training_sequences)
np.save(join(save_folder, "validation_sequences.npy"), validation_sequences)
np.save(join(save_folder, "training_targets.npy"), training_targets)
np.save(join(save_folder, "validation_targets.npy"), validation_targets)


# Create the LSTM model

batch_size = 14
num_epochs = 1000
# batches_per_print = 500
epochs_per_print = 500
torch.manual_seed(1)

train_seq_path = join(save_folder, "training_sequences.npy")
train_target_path = join(save_folder, "training_targets.npy")
val_seq_path = join(save_folder, "validation_sequences.npy")
val_target_path = join(save_folder, "validation_targets.npy")

training_dataset = NO2Dataset(train_seq_path, train_target_path)
validation_dataset = NO2Dataset(val_seq_path, val_target_path)

training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=1):
        super().__init__()  # runs init for the Module parent class
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)  # LSTM layers
        self.linear = nn.Linear(hidden_layer_size, output_size)  # Linear layers
        # Hidden cell variable contains previous hidden state and previous cell state.
        # self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
        #                     torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        # Pass input sequence through the lstm layer, which outputs the layer output, hidden state and cell state
        # at the current time step.

        lstm_out, hidden_state_cell_state = self.lstm(input_seq)#, self.hidden_cell)
        # print(lstm_out.shape, hidden_state_cell_state[0].shape, hidden_state_cell_state[1].shape)
        # Pass the lstm output to the linear layer, which calculates the dot product between
        # the layer input and weight matrix.
        prediction = self.linear(lstm_out[:, -1, :])  # We want the most recent hidden state
        # print(prediction.shape)
        return prediction.squeeze()


# Create model object of the LSTM class, define a loss function, define the optimiser.
model = LSTM(input_size=training_dataset.nfeatures(), hidden_layer_size=100)
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"Model:\n{model}")

# Train the LSTM model
filename = "lstm_model.tar"
save_path = join(load_folder, filename)

training_loss_history = []
validation_loss_history = []
# Currently no batches or validation set.

print("Begin training...")
for epoch in range(num_epochs):
    model.train()
    loss_sum = 0  # for storing
    # running_loss = 0.0  # for printing

    for batch_num, data in enumerate(training_dataloader):
        # print(batch_num)
        sequences_training = data["sequences"]
        targets_training = data["targets"]
        optimiser.zero_grad()
        # model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),  # Set to zero
        #                      torch.zeros(1, 1, model.hidden_layer_size))
        # Run the forward pass
        y_predict = model(sequences_training)
        # Compute the loss and gradients
        single_loss = criterion(y_predict, targets_training)
        single_loss.backward()
        # Update the parameters
        optimiser.step()

        # running_loss += single_loss.item()
        loss_sum += single_loss.item()*batch_size

    # Print the loss after every 25 epochs
    #     if batch_num % batches_per_print == 0:
    #         print(f"epoch: {epoch:3} batch {batch_num} loss: {running_loss/batches_per_print:10.8f}")
    #         running_loss = 0.0
    if epoch % epochs_per_print == 0 :
        print(f"Epoch {epoch} training loss: {loss_sum / len(training_dataset)}")
    training_loss_history.append(loss_sum / len(training_dataset))  # Save the training loss after every epoch
    # Validation set
    model.eval()
    validation_loss_sum = 0
    with torch.no_grad():
        for batch_num, data in enumerate(validation_dataloader):
            sequences_val = data["sequences"]
            targets_val = data["targets"]
            y_predict_validation = model(sequences_val)
            single_loss = criterion(y_predict_validation, targets_val)
            validation_loss_sum += single_loss.item()*batch_size
    # Store the model with smallest validation loss. Check if the validation loss is the lowest BEFORE
    # saving it to loss history (otherwise it will not be lower than itself)
    if (not validation_loss_history) or validation_loss_sum / len(validation_sequences) < min(validation_loss_history):
        best_model = deepcopy(model.state_dict())
        best_epoch = epoch
        # print(f"Epoch {epoch} validation loss: {validation_loss_sum / len(validation_sequences)}")
    validation_loss_history.append(validation_loss_sum / len(validation_sequences))  # Save the val loss every epoch.

        # Save the model after every 2 epochs
    if epoch % 2 == 0:
        torch.save({
            "total_epochs": epoch,
            "final_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "training_loss_history": training_loss_history,
            "best_state_dict": best_model,
            "best_epoch": best_epoch,
            "validation_loss_history": validation_loss_history
        }, save_path)

print(f"Finished training for {num_epochs} epochs.")

