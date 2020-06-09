import torch
import numpy as np
import pandas as pd
from os.path import join, dirname, realpath, exists
from os import listdir, makedirs
from dataset import NO2Dataset
from torch.utils.data import DataLoader
from lstm_model_class import LSTM
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import config

training_window = config.training_window  # consider the last X months of NO2 for each breast cancer diagnosis month
quantile_step = config.quantile_step # Make this False if not using.

ccgs = config.ccgs
#ccg = config.ccg
test_year = config.test_year
model_epoch = config.model_epoch  # Choose "final" or "best" model.

# One age category
age_category = config.age_category
print(f"{ccgs}\n{age_category}")

hidden_layer_size = config.hidden_layer_size
batch_size = config.batch_size
torch.manual_seed(config.random_seed)

if quantile_step:
    aggregation = f"{int(1/quantile_step)}_quantiles"
else:
    aggregation = "_".join(config.aggregation)
load_folder = join(join(join(dirname(realpath(__file__)), "_".join(ccgs)), aggregation), f"{training_window}_month_tw")



# Load train & test data
training_dataset = NO2Dataset(join(load_folder, "train_val_sequences.npy"), join(load_folder, f"train_val_targets_{age_category}.npy"))
test_dataset = NO2Dataset(join(load_folder, "test_sequences.npy"), join(load_folder, f"test_targets_{age_category}.npy"))

training_dataloader = DataLoader(training_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Load the model
filename = f"lstm_model_{age_category}_hl{hidden_layer_size}"
if config.noise_standard_deviation:
    filename += f"_augmented{config.noise_standard_deviation}".replace(".", "")
checkpoint = torch.load(join(load_folder, filename+".tar"))
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

with torch.no_grad():
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
train_mse = mean_squared_error(training_targets, training_prediction)
print(f"Train R sq {train_rsq}\nTrain MSE {train_mse}")

# Make predictions on test set
test_targets = []
test_prediction = []

with torch.no_grad():
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
test_mse = mean_squared_error(test_targets, test_prediction)
print(f"Test R sq {test_rsq}\nTest MSE {test_mse}")

# Map numpy arrays to NCRAS dataframe.
training_dates_ccgs = np.load(join(load_folder, "train_val_dates.npy"), allow_pickle=True)
test_dates_ccgs = np.load(join(load_folder, f"test_dates_{age_category}.npy"), allow_pickle=True)

training_df, test_df = pd.DataFrame(training_dates_ccgs, columns=["date", "ccg"]).set_index("date"), pd.DataFrame(test_dates_ccgs, columns=["date", "ccg"]).set_index("date")
training_df["target"], training_df["prediction"] = training_targets, training_prediction
test_df["target"], test_df["prediction"] = test_targets, test_prediction
training_df.index, test_df.index = pd.to_datetime(training_df.index), pd.to_datetime(test_df.index)
print(f"\nTraining dataframe {training_df.shape}, Test dataframe {test_df.shape}")
ccgs = test_df["ccg"].unique()
print(f"{len(ccgs)} CCGs in test set")

# Make plots
## Prediction plots
# train_dates = pd.date_range(f"2002-06", f"{test_year}-01", freq="M")
# test_dates = pd.date_range(f"{test_year}-01", f"{test_year+1}-01", freq="M")
# print(f"\nTrain dates {train_dates.shape}, Test dates {test_dates.shape}")
save_folder = join(load_folder, "results_plots")
if not exists(save_folder):
    makedirs(save_folder)
print("Plotting CCGs...")
for ccg in ccgs:
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    axs[0].plot(training_df.loc[training_df["ccg"] == ccg].index, training_df.loc[training_df["ccg"] == ccg, "target"], label="observed")
    axs[0].plot(training_df.loc[training_df["ccg"] == ccg].index, training_df.loc[training_df["ccg"] == ccg, "prediction"], label="prediction")
    axs[0].set_title(f"Training set (2002-06 to {test_year-1}-12)")
    axs[0].annotate(f"R$^2$ = {train_rsq}  MSE = {train_mse}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=12)
    axs[1].plot(test_df.loc[test_df["ccg"] == ccg].index, test_df.loc[test_df["ccg"] == ccg, "target"], label="observed")
    axs[1].plot(test_df.loc[test_df["ccg"] == ccg].index, test_df.loc[test_df["ccg"] == ccg, "prediction"], label="prediction")
    axs[1].set_title(f"Test set ({test_year})")
    axs[1].annotate(f"R$^2$ = {test_rsq}  MSE = {test_mse}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=12)

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

    ccg = ccg.replace("NHS ", "").replace(" ", "_")
    try:
        ccg = ccg[:ccg.index("_(")]
    except:
        pass

    plot_filename = f"{ccg}_timeseries_{age_category}_hl{hidden_layer_size}"
    if model_epoch == "final":
        plot_filename = f"{ccg}_timeseries_{age_category}_overfit_hl{hidden_layer_size}"
    if config.noise_standard_deviation:
        plot_filename += f"_augmented{config.noise_standard_deviation}".replace(".", "")
    fig.savefig(join(save_folder, plot_filename+".png"), dpi=fig.dpi)
    # plt.show()
    plt.close()
