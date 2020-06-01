import torch
import torch.nn as nn
from os.path import join, dirname, realpath
from copy import deepcopy
from dataset import NO2Dataset
from torch.utils.data import DataLoader

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
num_epochs = 1000
batches_per_print = False
epochs_per_print = 50
torch.manual_seed(1)

train_seq_path = join(load_folder, "training_sequences.npy")
train_target_path = join(load_folder, f"training_targets_{age_category}.npy")
val_seq_path = join(load_folder, "validation_sequences.npy")
val_target_path = join(load_folder, f"validation_targets_{age_category}.npy")

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
filename = f"lstm_model_{age_category}.tar"
save_path = join(load_folder, filename)

training_loss_history = []
validation_loss_history = []
# Currently no batches or validation set.

print("Begin training...")
for epoch in range(num_epochs):
    model.train()
    loss_sum = 0  # for storing
    running_loss = 0.0  # for printing

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

        running_loss += single_loss.item()
        loss_sum += single_loss.item()*batch_size

    # Print the loss after every X batches or after every Y epochs
        if batches_per_print and batch_num % batches_per_print == 0:
            print(f"epoch: {epoch:3} batch {batch_num} loss: {running_loss/batches_per_print:10.8f}")
            running_loss = 0.0
    if epochs_per_print and epoch % epochs_per_print == 0 :
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
    if (not validation_loss_history) or validation_loss_sum / len(validation_dataset) < min(validation_loss_history):
        best_model = deepcopy(model.state_dict())
        best_epoch = epoch
        # print(f"Epoch {epoch} validation loss: {validation_loss_sum / len(validation_sequences)}")
    validation_loss_history.append(validation_loss_sum / len(validation_dataset))  # Save the val loss every epoch.

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

