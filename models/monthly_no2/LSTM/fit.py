import torch
import torch.nn as nn
from os.path import join, dirname, realpath
from copy import deepcopy
from dataset import NO2Dataset
from torch.utils.data import DataLoader
from lstm_model_class import LSTM
import config

training_window = config.training_window  # consider the last X months of NO2 for each breast cancer diagnosis month
quantile_step = config.quantile_step  # Make this False if not using.

ccgs = config.ccgs
ccg = config.ccg

# One age category
age_category = config.age_category
print(f"{ccg}\n{age_category}\n{training_window}-month training window")

hidden_layer_size = config.hidden_layer_size
batch_size = config.batch_size
num_epochs = config.num_epochs
batches_per_print = config.batches_per_print
epochs_per_print = config.epochs_per_print
torch.manual_seed(config.random_seed)

if quantile_step:
    aggregation = f"{int(1/quantile_step)}_quantiles"
else:
    aggregation = "_".join(config.aggregation)
print(aggregation)
load_folder = join(join(join(dirname(realpath(__file__)), ccg), aggregation), f"{training_window}_month_tw")

train_seq_path = join(load_folder, "training_sequences.npy")
train_target_path = join(load_folder, f"training_targets_{age_category}.npy")
val_seq_path = join(load_folder, "validation_sequences.npy")
val_target_path = join(load_folder, f"validation_targets_{age_category}.npy")

training_dataset = NO2Dataset(train_seq_path, train_target_path, noise_std=config.noise_standard_deviation)
validation_dataset = NO2Dataset(val_seq_path, val_target_path)

training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

if config.compute_test_loss:
    test_dataset = NO2Dataset(join(load_folder, "test_sequences.npy"), join(load_folder, f"test_targets_{age_category}.npy"))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
test_loss_history = []

# Create model object of the LSTM class, define a loss function, define the optimiser.
model = LSTM(input_size=training_dataset.nfeatures(), hidden_layer_size=hidden_layer_size)
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
print(f"Model:\n{model}")

# Train the LSTM model
filename = f"lstm_model_{age_category}_hl{hidden_layer_size}"
if config.noise_standard_deviation:
    filename += "_augmented"
save_path = join(load_folder, filename+".tar")

training_loss_history = []
validation_loss_history = []

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
        loss_sum += single_loss.item()*data["targets"].shape[0]  # Account for different batch size with final batch

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
            validation_loss_sum += single_loss.item()*data["targets"].shape[0]
        if config.compute_test_loss:
            test_loss_sum = 0
            for batch_num, data in enumerate(test_dataloader):
                sequences_test = data["sequences"]
                targets_test = data["targets"]
                y_predict_test = model(sequences_test)
                single_loss = criterion(y_predict_test, targets_test)
                test_loss_sum += single_loss.item()*data["targets"].shape[0]
            test_loss_history.append(test_loss_sum / len(test_dataset))
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
            "validation_loss_history": validation_loss_history,
            "test_loss_history": test_loss_history
        }, save_path)

print(f"Finished training for {num_epochs} epochs.")

