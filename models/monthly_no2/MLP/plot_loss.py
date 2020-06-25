import torch
from os.path import join, dirname, realpath
from dataset import NO2Dataset
from mlp_class import MLP
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import config

# Define these from the config file
training_window = config.training_window
quantile_step = config.quantile_step
ccgs = config.ccgs
test_year = config.test_year
age_category = config.age_category
print(f"{ccgs}\n{age_category}")
age_category = age_category.replace("<", "").replace(">=", "")
hidden_layer_sizes = config.hidden_layer_sizes
batch_size = config.batch_size
torch.manual_seed(config.random_seed)

# Determine the appropriate monthly aggregation statistic for NO2
if quantile_step:
    aggregation = f"{int(1/quantile_step)}_quantiles"
else:
    aggregation = "_".join(config.aggregation)

# Define the loading folder for the experiment
load_folder = join(dirname(realpath(__file__)), "_".join(ccgs), aggregation, f"{training_window}_month_tw")

if ccgs == ["clustered_ccgs"]:
    label = f"cluster_{config.cluster_label}of{config.n_clusters}"
    load_folder = join(dirname(realpath(__file__)), ccgs[0], label, aggregation, f"{training_window}_month_tw")

# Load training data so we can compute the number of features
training_dataset = NO2Dataset(join(load_folder, "training_sequences.npy"), join(load_folder, f"training_targets_{age_category}.npy"))

# Load the model
filename = f"mlp_model_{age_category}_{len(hidden_layer_sizes)}hls"
if config.noise_standard_deviation:
    filename += f"_augmented{config.noise_standard_deviation}".replace(".", "")

checkpoint = torch.load(join(load_folder, filename+".tar"))
model = MLP(h_sizes=config.hidden_layer_sizes, out_size=config.out_size)

# Load the required info for plotting losses
training_losses = checkpoint["training_loss_history"]
val_losses = checkpoint["validation_loss_history"]
best_epoch = checkpoint["best_epoch"]
epochs = checkpoint["total_epochs"]
# Also load the test set losses if these were computed
if config.compute_test_loss:
    test_losses = checkpoint["test_loss_history"]

# Plot training and validation loss history and annotate the epoch with best validation loss

# Initiate the plot figure and define the filename for saving
fig, ax = plt.subplots(figsize=(12, 8))
save_name = f"loss_history_{age_category}_{len(hidden_layer_sizes)}hls"

# Plot training and validation loss histories
ax.plot(range(epochs+1), training_losses, label="training loss", alpha=0.8)
ax.plot(range(epochs+1), val_losses, label="validation loss", alpha=0.8)

# Plot test loss history if computed; edit the filename for saving
if config.compute_test_loss:
    save_name += "_withtest"
    ax.plot(range(epochs+1), test_losses, label="test loss", alpha=0.8)

# Edit the filename for saving the plot if data augmentation was used
if config.noise_standard_deviation:
    save_name += f"_augmented{config.noise_standard_deviation}".replace(".", "")

# Add a point labelling the epoch with lowest validation loss
ax.scatter(best_epoch, min(val_losses))

# Add a legend and title, label the plot and axes
plt.annotate(f"epoch {best_epoch}", (best_epoch*1.05, min(val_losses)))
plt.legend()
plt.xlabel("epoch")
plt.ylabel("MSE loss")
ccgs = ", ".join(ccgs)
plt.title(f"mlp training loss for {ccgs}")

# plt.show()
fig.tight_layout()

# Save the plot as a PNG file
fig.savefig(join(load_folder, save_name+".png"), dpi=fig.dpi)
