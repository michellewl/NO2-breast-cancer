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
model_epoch = "best"  # Choose "final" or "best" model.

# One age category
age_category = "all_ages"
print(f"{ccg}\n{age_category}")

hidden_layer_size = 100
batch_size = 14
torch.manual_seed(1)

if quantile_step:
    aggregation = f"{int(1/quantile_step)}_quantiles"
load_folder = join(join(join(dirname(realpath(__file__)), ccg), aggregation), f"{training_window}_month_tw")



# Load train & test data
training_dataset = NO2Dataset(join(load_folder, "train_val_sequences.npy"), join(load_folder, f"train_val_targets_{age_category}.npy"))
test_dataset = NO2Dataset(join(load_folder, "test_sequences.npy"), join(load_folder, f"test_targets_{age_category}.npy"))

training_dataloader = DataLoader(training_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Load the model
checkpoint = torch.load(join(load_folder,  f"lstm_model_{age_category}.tar"))
model = LSTM(input_size=training_dataset.nfeatures(), hidden_layer_size=hidden_layer_size)
print(model)

if model_epoch == "best":
    model.load_state_dict(checkpoint["best_state_dict"])
    epoch = checkpoint["best_epoch"]
elif model_epoch == "final":
    model.load_state_dict(checkpoint["final_state_dict"])
    epoch = checkpoint["total_epochs"]


model.eval()

# Make predictions on training set
training_targets = []
training_prediction = []

for batch_num, data in enumerate(training_dataloader):
    sequences = data["sequences"]
    targets = data["targets"]

    outputs = model(sequences)

    training_targets.append(targets.detach().numpy())
    training_prediction.append(outputs.detach().numpy())

y_normaliser = joblib.load(join(load_folder, f"y_{age_category}_normaliser.sav"))
training_targets = y_normaliser.inverse_transform(np.concatenate(training_targets, axis=None))
training_prediction = y_normaliser.inverse_transform(np.concatenate(training_prediction, axis=None))
print(f"Train targets {training_targets.shape}, Train predict {training_prediction.shape}")
train_rsq = r2_score(training_targets, training_prediction)
print(f"Train R sq {train_rsq}")

# Make predictions on test set
test_targets = []
test_prediction = []

for batch_num, data in enumerate(test_dataloader):
    sequences = data["sequences"]
    targets = data["targets"]

    outputs = model(sequences)

    test_targets.append(targets.detach().numpy())
    test_prediction.append(outputs.detach().numpy())

test_targets = y_normaliser.inverse_transform(np.concatenate(test_targets, axis=None))
test_prediction = y_normaliser.inverse_transform(np.concatenate(test_prediction, axis=None))
print(f"\nTest targets {test_targets.shape}, Test predict {test_prediction.shape}")
test_rsq = r2_score(test_targets, test_prediction)
print(f"Test R sq {test_rsq}")

# Make plots
## Prediction plots
train_dates = pd.date_range(f"2002-06", f"{test_year}-01", freq="M")
test_dates = pd.date_range(f"{test_year}-01", f"{test_year+1}-01", freq="M")
print(f"\nTrain dates {train_dates.shape}, Test dates {test_dates.shape}")

fig, axs = plt.subplots(2, 1, figsize=(15, 10))

axs[0].plot(train_dates, training_targets, label="observed")
axs[0].plot(train_dates, training_prediction, label="predicted")
axs[0].set_title(f"Training set (2002-06 to {test_year-1}-12)")
axs[0].annotate(f"R$^2$ = {train_rsq}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=12)
axs[1].plot(test_dates, test_targets, label="observed")
axs[1].plot(test_dates, test_prediction, label="prediction")
axs[1].set_title(f"Test set ({test_year})")
axs[1].annotate(f"R$^2$ = {test_rsq}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=12)

for ax in axs.flatten():
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Breast cancer cases ({age_category.replace( '_', ' ')}) per capita")

fig.suptitle(f"LSTM model for {ccg}")

plt.figtext(0.1, 0.5, f"{training_window} month training window",
            fontsize=12)
plt.figtext(0.1, 0.48, f"LSTM hidden layer size {model.hidden_layer_size}", fontsize=12)
plt.figtext(0.1, 0.46, f"Model learnt at epoch {epoch}", fontsize=12)

plt.legend(loc=1)
fig.subplots_adjust(top=0.5)
fig.tight_layout(pad=2)

if model_epoch == "best":
    plot_filename = f"time_series_{age_category}_hl{hidden_layer_size}.png"
elif model_epoch == "final":
    plot_filename = f"time_series_{age_category}_overfit_hl{hidden_layer_size}.png"
fig.savefig(join(load_folder, plot_filename), dpi=fig.dpi)
# plt.show()



