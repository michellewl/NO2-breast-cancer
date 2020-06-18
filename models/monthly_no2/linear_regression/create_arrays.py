import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import join, dirname, realpath, exists
import re
from sklearn.preprocessing import StandardScaler
import joblib
import config
from dateutil.relativedelta import relativedelta
from functions import create_x_sequences



# Define these from the config file
training_window = config.training_window
quantile_step = config.quantile_step
ccgs = config.ccgs
age_category = config.age_category
print(f"{ccgs}\n{age_category}")
if training_window:
    input_description = f"{training_window}-month_tw"
else:
    input_description = "paired_in_out"
print(input_description)
age_category_rename = age_category.replace("<", "").replace(">=", "")  # Can't have < or > in filepaths
test_year = config.test_year


# Determine the appropriate monthly aggregation statistics for NO2
if quantile_step:
    aggregation = [f"{int(method*100)}_quantile" for method in np.round(np.arange(0, 1+quantile_step, quantile_step), 2).tolist()]
else:
    aggregation = config.aggregation
print(aggregation)


no2_folder = join(dirname(dirname(dirname(dirname(realpath(__file__))))), "data", "LAQN", f"{config.laqn_start_date}_{config.laqn_end_date}", "monthly")
no2_filenames = [file for method in aggregation for file in listdir(no2_folder) if re.findall(f"ccgs_monthly_{method}.csv", file)]
print(no2_filenames)

ncras_folder = join(dirname(dirname(dirname(dirname(realpath(__file__))))), "data", "NCRAS")
ncras_filename = [f for f in listdir(ncras_folder) if "ccgs_population_fraction.csv" in f][0]
print(ncras_filename)

# Load the NCRAS breast cancer dataframe
ncras_df = pd.read_csv(join(ncras_folder, ncras_filename)).set_index("date")

# Load the clustered CCG dataframe
cluster_filename = join(dirname(dirname(dirname(dirname(realpath(__file__))))), "data", "exploration",
                        f"{config.cluster_variables}_{config.n_clusters}_clusters_2013-2018.csv")
cluster_df = pd.read_csv(cluster_filename)

# Determine the appropriate save folder and create it if it doesn't exist
if quantile_step:
    aggregation = [str(len(aggregation)-1), "quantiles"]

if len(ccgs) > 1:
    save_folder = join(dirname(realpath(__file__)), "_".join(ccgs), "_".join(aggregation))
elif ccgs == ["clustered_ccgs"]:
    label = f"cluster_{config.cluster_label}of{config.n_clusters}"
    save_folder = join(dirname(realpath(__file__)), ccgs[0], label, "_".join(aggregation))
else:
    save_folder = join(dirname(realpath(__file__)), ccgs[0], "_".join(aggregation))

save_folder = join(save_folder, input_description)

if not exists(save_folder):
    makedirs(save_folder)

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

# # Use the assigned training window length to define the start and end dates for training and testing
ncras_start_date, ncras_end_date = ncras_df.index.min(), ncras_df.index.max()

if training_window:
    no2_start_training_date, no2_end_training_date = ncras_start_date - relativedelta(months=training_window), \
                                   ncras_end_date - relativedelta(months=1, years=1)
    no2_start_test_date, no2_end_test_date = ncras_end_date - relativedelta(months=training_window, years=1), \
                                             ncras_end_date - relativedelta(months=1)
else:
    no2_start_training_date, no2_end_training_date = ncras_start_date, \
                                                     ncras_end_date - relativedelta(years=1)
    no2_start_test_date, no2_end_test_date = ncras_end_date - relativedelta(months=11), \
                                             ncras_end_date

# Initiate lists for arrays
# Train
training_inputs_list = []
training_targ_list = []
# Test
test_inputs_list = []
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

    if training_window:
        # Create the input sequences
        x_train = create_x_sequences(x_train, len(y_train), training_window)
        x_test = create_x_sequences(x_test, len(y_test), training_window)

    # Add all the arrays to their relevant lists
    training_inputs_list.append(x_train)
    training_targ_list.append(y_train)
    test_inputs_list.append(x_test)
    test_targ_list.append(y_test)


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
training_inputs = np.concatenate(training_inputs_list, axis=0)
training_targets = np.concatenate(training_targ_list, axis=0)

# Test arrays
test_inputs = np.concatenate(test_inputs_list, axis=0)
test_targets = np.concatenate(test_targ_list, axis=0)

# Dates and CCGs arrays (for plotting/evaluation)
training_dates = np.concatenate(training_meta_list, axis=0)
test_dates = np.concatenate(test_meta_list, axis=0)

# Normalise all the arrays
print(training_inputs.shape)

# Fit the normaliser to training set. StandardScaler can only take up to 2 dimensions.
if training_window:
    x_normaliser = StandardScaler().fit(training_inputs.reshape(-1, training_inputs.shape[2]))
else:
    x_normaliser = StandardScaler().fit(training_inputs)

y_normaliser = StandardScaler().fit(training_targets)

# Save to later apply un-normalisation to test sets for plotting/evaluation
#joblib.dump(x_normaliser, join(save_folder, "x_normaliser.sav"))
joblib.dump(y_normaliser, join(save_folder, f"y_{age_category_rename}_normaliser.sav"))

# Normalise input and output data
if training_window:
    training_inputs = x_normaliser.transform(training_inputs.reshape(-1, training_inputs.shape[2])).reshape(training_inputs.shape)
else:
    training_inputs = x_normaliser.transform(training_inputs)
training_targets = y_normaliser.transform(training_targets)

if training_window:
    test_inputs = x_normaliser.transform(test_inputs.reshape(-1, test_inputs.shape[2])).reshape(test_inputs.shape)
else:
    test_inputs = x_normaliser.transform(test_inputs)
test_targets = y_normaliser.transform(test_targets)


print(f"\nDropping NaNs\nTraining {np.isnan(training_inputs).any(axis=1).sum()}\n"
      f"Test {np.isnan(test_inputs).any(axis=1).sum()}")

# Look along dimensions 1 & 2 for NaNs

if training_window:
    drop_nan_axes = (1, 2)
else:
    drop_nan_axes = 1

training_inputs_dropna = training_inputs[np.logical_not(np.isnan(training_inputs).any(axis=drop_nan_axes))]
training_targets_dropna = training_targets[np.logical_not(np.isnan(training_inputs).any(axis=drop_nan_axes))]

test_inputs_dropna = test_inputs[np.logical_not(np.isnan(test_inputs).any(axis=drop_nan_axes))]
test_targets_dropna = test_targets[np.logical_not(np.isnan(test_inputs).any(axis=drop_nan_axes))]

training_dates_dropna = training_dates[np.logical_not(np.isnan(training_inputs).any(axis=drop_nan_axes))]
test_dates_dropna = test_dates[np.logical_not(np.isnan(test_inputs).any(axis=drop_nan_axes))]


print(f"\nTraining inputs {training_inputs_dropna.shape} Training targets {training_targets_dropna.shape}")
print(f"Test inputs {test_inputs_dropna.shape} Test targets {test_targets_dropna.shape}")

print(f"Training dates {training_dates_dropna.shape} Test dates {test_dates_dropna.shape}")

# Save the arrays
age_category = age_category_rename  # Can't have < or > in filepaths
np.save(join(save_folder, f"training_inputs.npy"), training_inputs_dropna)
np.save(join(save_folder, f"training_targets_{age_category}.npy"), training_targets_dropna)
np.save(join(save_folder, f"test_inputs.npy"), test_inputs_dropna)
np.save(join(save_folder, f"test_targets_{age_category}.npy"), test_targets_dropna)

np.save(join(save_folder, f"training_dates.npy"), training_dates_dropna)
np.save(join(save_folder, f"test_dates_{age_category}.npy"), test_dates_dropna)

print("\nSaved npy arrays.")
