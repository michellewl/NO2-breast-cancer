import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import join, dirname, realpath, exists
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from dateutil.relativedelta import relativedelta
import config
from functions import create_x_sequences

# Define these from the config file
laqn_start_date = config.laqn_start_date
laqn_end_date = config.laqn_end_date

training_window = config.training_window
quantile_step = config.quantile_step

ccgs = config.ccgs
age_category = config.age_category
print(f"{ccgs}\n{age_category}")
test_year = config.test_year

# Determine the appropriate monthly aggregation statistics for NO2
if quantile_step:
    aggregation = [f"{int(method*100)}_quantile" for method in np.round(np.arange(0, 1+quantile_step, quantile_step), 2).tolist()]
else:
    aggregation = config.aggregation
print(aggregation)

# Locate the data files required
no2_folder = join(dirname(dirname(dirname(dirname(realpath(__file__))))), "data", "LAQN", f"{laqn_start_date}_{laqn_end_date}", "monthly")
no2_filenames = [file for method in aggregation for file in listdir(no2_folder) if re.findall(f"ccgs_monthly_{method}.csv", file)]
print(no2_filenames)

ncras_folder = join(dirname(dirname(dirname(dirname(realpath(__file__))))), "data", "NCRAS")
ncras_filename = [f for f in listdir(ncras_folder) if "ccgs_population_fraction.csv" in f][0]
print(ncras_filename)


# Determine the appropriate save folder and create it if it doesn't exist
if quantile_step:
    aggregation = [str(len(aggregation)-1), "quantiles"]

if len(ccgs) > 1:
    save_folder = join(dirname(realpath(__file__)), "_".join(ccgs), "_".join(aggregation), f"{training_window}_month_tw")
elif ccgs == ["clustered_ccgs"]:
    label = f"cluster_{config.cluster_label}of{config.n_clusters}"
    save_folder = join(dirname(realpath(__file__)), ccgs[0], label, "_".join(aggregation),
                       f"{training_window}_month_tw")
else:
    save_folder = join(dirname(realpath(__file__)), ccgs[0], "_".join(aggregation),
                       f"{training_window}_month_tw")
if not exists(save_folder):
    makedirs(save_folder)

# Load the NCRAS breast cancer dataframe
ncras_df = pd.read_csv(join(ncras_folder, ncras_filename)).set_index("date")

# Load the clustered CCG dataframe
cluster_filename = join(dirname(dirname(dirname(dirname(realpath(__file__))))), "data", "exploration",
                        f"{config.cluster_variables}_{config.n_clusters}_clusters_2013-2018.csv")
cluster_df = pd.read_csv(cluster_filename)

# Define the list of CCGs to be used in the analysis
if ccgs == ["all_ccgs"]:
    ccgs = ncras_df["ccg_name"].unique().tolist()
elif ccgs == ["clustered_ccgs"]:
    ccgs = cluster_df.loc[cluster_df["cluster_label"] == config.cluster_label, "ccg"].unique().tolist()
print(f"{len(ccgs)} CCGs.")
print(ccgs)

# Trim the NCRAS dataframe to the required CCGs
ncras_df = ncras_df.loc[ncras_df["ccg_name"].isin(ccgs)]
ncras_df.index = pd.to_datetime(ncras_df.index)

# Use the assigned training window length to define the start and end dates for training and testing
ncras_start_date, ncras_end_date = ncras_df.index.min(), ncras_df.index.max()
no2_start_training_date, no2_end_training_date = ncras_start_date - relativedelta(months=training_window), \
                               ncras_end_date - relativedelta(months=1, years=1)
no2_start_test_date, no2_end_test_date = ncras_end_date - relativedelta(months=training_window, years=1), \
                                         ncras_end_date - relativedelta(months=1)

# Initiate lists for arrays
# All training arrays, including validation set (used for plotting/evaluation)
train_val_seq_list = []
train_val_targ_list = []
# Training arrays excluding validation set
train_seq_list =[]
train_targ_list = []
# Validation arrays
val_seq_list =[]
val_targ_list = []
# Test set arrays
test_seq_list = []
test_targ_list = []
# Dates and CCG names for training and test sets (needed for plotting/evaluation)
training_meta_list = []
test_meta_list =[]

print("Processing CCGs...")
for ccg in ccgs:

    # Load the split the NO2 data
    no2_training_array_list = []
    no2_test_array_list = []
    for no2_file in no2_filenames:
        no2_df = pd.read_csv(join(no2_folder, no2_file)).set_index("MeasurementDateGMT")
        no2_df.index = pd.to_datetime(no2_df.index)
        no2_df = no2_df.loc[:, ccgs]

        no2_array = no2_df.loc[(no2_df.index >= no2_start_training_date) & (no2_df.index <= no2_end_training_date),
                               ccg].values.reshape(-1, 1)
        no2_training_array_list.append(no2_array)

        no2_array = no2_df.loc[(no2_df.index >= no2_start_test_date) &
                                              (no2_df.index <= no2_end_test_date), ccg].values.reshape(-1, 1)
        no2_test_array_list.append(no2_array)

    # Join the NO2 data arrays for training and testing
    x_train = np.concatenate(no2_training_array_list, axis=1)
    x_test = np.concatenate(no2_test_array_list, axis=1)

    # Define the breast cancer data arrays for training and testing
    y_train = ncras_df.loc[(ncras_df.index.year != test_year) & (ncras_df.ccg_name == ccg), age_category] \
        .values.reshape(-1, 1)
    y_test = ncras_df.loc[(ncras_df.index.year == test_year) & (ncras_df.ccg_name == ccg), age_category] \
        .values.reshape(-1, 1)

    # Fit the normaliser
    x_normaliser = StandardScaler().fit(x_train)
    y_normaliser = StandardScaler().fit(y_train)
    # Save to later apply un-normalisation to test sets for plotting/evaluation
    joblib.dump(x_normaliser, join(save_folder, "x_normaliser.sav"))
    joblib.dump(y_normaliser, join(save_folder, f"y_{age_category}_normaliser.sav"))
    # Normalise input and output data
    x_train_norm = x_normaliser.transform(x_train)
    y_train_norm = y_normaliser.transform(y_train).squeeze()
    x_test_norm = x_normaliser.transform(x_test)
    y_test_norm = y_normaliser.transform(y_test).squeeze()

    # Create the input sequences for the LSTM
    train_val_inputs = create_x_sequences(x_train_norm, len(y_train_norm), training_window)
    test_inputs = create_x_sequences(x_test_norm, len(y_test_norm), training_window)

    # Split the training and validation sets
    validation_size = 0.2
    training_sequences, validation_sequences, training_targets, validation_targets = train_test_split(train_val_inputs, y_train_norm, test_size=validation_size, random_state=1)

    # Add all the arrays to their relevant lists
    train_seq_list.append(training_sequences)
    train_targ_list.append(training_targets)
    val_seq_list.append(validation_sequences)
    val_targ_list.append(validation_targets)
    test_seq_list.append(test_inputs)
    test_targ_list.append(y_test_norm)
    train_val_seq_list.append(train_val_inputs)
    train_val_targ_list.append(y_train_norm)

    # Determine the date & CCG arrays and append to the relevant lists
    training_date_array = ncras_df.loc[(ncras_df.index.year != test_year) & (ncras_df.ccg_name == ccg), age_category].index.map(str).to_numpy().reshape(-1, 1)
    test_date_array = ncras_df.loc[(ncras_df.index.year == test_year) & (ncras_df.ccg_name == ccg), age_category].index.map(str).to_numpy().reshape(-1, 1)
    ccg_train_array = np.repeat(np.array([ccg]), training_date_array.shape[0]).reshape(-1, 1)
    ccg_test_array = np.repeat(np.array([ccg]), test_date_array.shape[0]).reshape(-1, 1)
    training_meta_array = np.concatenate((training_date_array, ccg_train_array), axis=1)
    test_meta_array = np.concatenate((test_date_array, ccg_test_array), axis=1)
    training_meta_list.append(training_meta_array)
    test_meta_list.append(test_meta_array)

# Concatentate the full arrays from the lists of arrays
# Training arrays
training_sequences = np.concatenate(train_seq_list, axis=0)
training_targets = np.concatenate(train_targ_list, axis=0)
# Validation arrays
validation_sequences = np.concatenate(val_seq_list, axis=0)
validation_targets = np.concatenate(val_targ_list, axis=0)
# Test arrays
test_sequences = np.concatenate(test_seq_list, axis=0)
test_targets = np.concatenate(test_targ_list, axis=0)
# Training + validation arrays (for plotting/evaluation)
train_val_sequences = np.concatenate(train_val_seq_list, axis=0)
train_val_targets = np.concatenate(train_val_targ_list, axis=0)
# Dates and CCGs arrays (for plotting/evaluation)
training_dates = np.concatenate(training_meta_list, axis=0)
test_dates = np.concatenate(test_meta_list, axis=0)

print(f"\nDropping NaNs\nTraining {np.isnan(training_sequences).any(axis=(1, 2)).sum()}\n"
      f"Validation {np.isnan(validation_sequences).any(axis=(1, 2)).sum()}\n"
            f"Train/val {np.isnan(train_val_sequences).any(axis=(1, 2)).sum()}\n"
      f"Test {np.isnan(test_sequences).any(axis=(1, 2)).sum()}")

# Look along dimensions 1 & 2 for NaNs
training_sequences_dropna = training_sequences[np.logical_not(np.isnan(training_sequences).any(axis=(1, 2)))]
training_targets_dropna = training_targets[np.logical_not(np.isnan(training_sequences).any(axis=(1, 2)))]
validation_sequences_dropna = validation_sequences[np.logical_not(np.isnan(validation_sequences).any(axis=(1, 2)))]
validation_targets_dropna = validation_targets[np.logical_not(np.isnan(validation_sequences).any(axis=(1, 2)))]
train_val_sequences_dropna = train_val_sequences[np.logical_not(np.isnan(train_val_sequences).any(axis=(1, 2)))]
train_val_targets_dropna = train_val_targets[np.logical_not(np.isnan(train_val_sequences).any(axis=(1, 2)))]
test_sequences_dropna = test_sequences[np.logical_not(np.isnan(test_sequences).any(axis=(1, 2)))]
test_targets_dropna = test_targets[np.logical_not(np.isnan(test_sequences).any(axis=(1, 2)))]
training_dates_dropna = training_dates[np.logical_not(np.isnan(train_val_sequences).any(axis=(1, 2)))]
test_dates_dropna = test_dates[np.logical_not(np.isnan(test_sequences).any(axis=(1, 2)))]


print(f"\nTraining sequences {training_sequences_dropna.shape} Training targets {training_targets_dropna.shape}")
print(f"Validation sequences {validation_sequences_dropna.shape} Validation targets {validation_targets_dropna.shape}")
print(f"Train/val sequences {train_val_sequences_dropna.shape} Train/val targets {train_val_targets_dropna.shape}")
print(f"Test sequences {test_sequences_dropna.shape} Test targets {test_targets_dropna.shape}")

print(f"Training dates {training_dates_dropna.shape} Test dates {test_dates_dropna.shape}")

# Save the arrays
np.save(join(save_folder, "training_sequences.npy"), training_sequences_dropna)
np.save(join(save_folder, "validation_sequences.npy"), validation_sequences_dropna)
np.save(join(save_folder, f"training_targets_{age_category}.npy"), training_targets_dropna)
np.save(join(save_folder, f"validation_targets_{age_category}.npy"), validation_targets_dropna)
np.save(join(save_folder, "train_val_sequences.npy"), train_val_sequences_dropna)
np.save(join(save_folder, f"train_val_targets_{age_category}.npy"), train_val_targets_dropna)

np.save(join(save_folder, "test_sequences.npy"), test_sequences_dropna)
np.save(join(save_folder, f"test_targets_{age_category}.npy"), test_targets_dropna)

np.save(join(save_folder, "train_val_sequences.npy"), train_val_sequences_dropna)
np.save(join(save_folder, f"train_val_targets_{age_category}.npy"), train_val_targets_dropna)

np.save(join(save_folder, "train_val_dates.npy"), training_dates_dropna)
np.save(join(save_folder, f"test_dates_{age_category}.npy"), test_dates_dropna)

print("\nSaved npy arrays.")
