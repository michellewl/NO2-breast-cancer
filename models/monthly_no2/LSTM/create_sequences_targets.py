import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import join, dirname, realpath, exists
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from dateutil.relativedelta import relativedelta

training_window = 3  # consider the last X months of NO2 for each breast cancer diagnosis month

# aggregation = ["min", "max"]
# aggregation = ["mean"]
quantile_step = 0.1  # Make this False if not using.

ccgs = ["NHS Central London (Westminster)", "NHS Richmond"]
ccg = ccgs[0]
test_year = 2017

dates_as_inputs = False

if quantile_step:
    aggregation = [f"{int(method*100)}_quantile" for method in np.round(np.arange(0, 1+quantile_step, quantile_step), 2).tolist()]
print(aggregation)

no2_folder = join(join(join(dirname(dirname(dirname(dirname(realpath(__file__))))), "data"), "LAQN"), "monthly")
no2_filenames = [file for method in aggregation for file in listdir(no2_folder) if re.findall(f"ccgs_monthly_{method}.csv", file)]
print(no2_filenames)

ncras_folder = join(join(dirname(dirname(dirname(dirname(realpath(__file__))))), "data"), "NCRAS")
ncras_filename = [f for f in listdir(ncras_folder) if "ccgs_population_fraction.csv" in f][0]
# print(ncras_filename)

ncras_df = pd.read_csv(join(ncras_folder, ncras_filename)).set_index("ccg_name").loc[ccgs]
ncras_df.reset_index(inplace=True)
ncras_df.set_index("date", inplace=True)
ncras_df.index = pd.to_datetime(ncras_df.index)

# print(ncras_df)

ncras_start_date, ncras_end_date = ncras_df.index.min(), ncras_df.index.max()
no2_start_training_date, no2_end_training_date = ncras_start_date - relativedelta(months=training_window), \
                               ncras_end_date - relativedelta(months=1, years=1)
no2_start_test_date, no2_end_test_date = ncras_end_date - relativedelta(months=training_window, years=1), ncras_end_date - relativedelta(months=1)
# print(ncras_start_date, no2_start_training_date, no2_end_training_date)
# print(ncras_end_date, no2_start_test_date, no2_end_test_date)


###################### NO2 PROCESSING ##########################
no2_df_list = []

for no2_file in no2_filenames:
    no2_df = pd.read_csv(join(no2_folder, no2_file)).set_index("MeasurementDateGMT")
    no2_df.index = pd.to_datetime(no2_df.index)
    no2_df = no2_df.loc[:, ccgs]
    no2_df_list.append(no2_df)
# print(no2_df_list)

no2_training_array_list = []
no2_test_array_list = []

if dates_as_inputs:
    input_dates_df = pd.DataFrame(index=no2_df_list[0].index)
    input_dates_df["year"] = input_dates_df.index.year
    input_dates_df["month_sin"] = np.sin(np.deg2rad((360/12) * input_dates_df.index.month))
    input_dates_df["month_cos"] = np.cos(np.deg2rad((360/12) * input_dates_df.index.month))
    training_dates_array = input_dates_df.loc[(input_dates_df.index >= no2_start_training_date) &
                                              (input_dates_df.index <= no2_end_training_date),
                                              ("year", "month_sin", "month_cos")].values.reshape(-1, 3)
    test_dates_array = input_dates_df.loc[(input_dates_df.index >= no2_start_test_date) &
                                              (input_dates_df.index <= no2_end_test_date), ("year", "month_sin", "month_cos")].values.reshape(-1, 3)
    # print(training_dates_array)

    no2_training_array_list.append(training_dates_array)
    no2_test_array_list.append(test_dates_array)

for no2_df in no2_df_list:
    no2_array = no2_df.loc[(no2_df.index >= no2_start_training_date) & (no2_df.index <= no2_end_training_date),
                           ccg].values.reshape(-1, 1)
    no2_training_array_list.append(no2_array)

    no2_array = no2_df.loc[(no2_df.index >= no2_start_test_date) &
                                          (no2_df.index <= no2_end_test_date), ccg].values.reshape(-1, 1)
    no2_test_array_list.append(no2_array)
# print(no2_training_array_list)
# print(no2_test_array_list)

age_categories = [col for col in ncras_df.columns if "age" in col]

# One CCG, one age category
age_category = age_categories[-1]
print(f"{ccg}\n{age_category}")

# Get data arrays and split x and y into train and test (prediction) sets.
x_train = np.concatenate(no2_training_array_list, axis=1)
x_test = np.concatenate(no2_test_array_list, axis=1)

# print(x_train)

y_train = ncras_df.loc[(ncras_df.index.year != test_year) & (ncras_df.ccg_name == ccg), age_category] \
    .values.reshape(-1, 1)
y_test = ncras_df.loc[(ncras_df.index.year == test_year) & (ncras_df.ccg_name == ccg), age_category] \
    .values.reshape(-1, 1)

# print(f"x train: {x_train.shape}"
#       f"\ny train: {y_train.shape}"
#       f"\nx test: {x_test.shape}"
#       f"\ny test: {y_test.shape}")

# Save folder
if quantile_step:
    aggregation = [str(len(aggregation)-1), "quantiles"]

save_folder = join(join(join(dirname(realpath(__file__)), ccg), "_".join(aggregation)), f"{training_window}_month_tw")
if not exists(save_folder):
    makedirs(save_folder)

# Normaliser
x_normaliser = StandardScaler().fit(x_train)
y_normaliser = StandardScaler().fit(y_train)
# Save normalisation to later apply to test sets
joblib.dump(x_normaliser, join(save_folder, "x_normaliser.sav"))
joblib.dump(y_normaliser, join(save_folder, f"y_{age_category}_normaliser.sav"))
# print(f"Saved normaliser to {save_folder}")

# Normalise input and output data
x_train_norm = x_normaliser.transform(x_train)
y_train_norm = y_normaliser.transform(y_train).squeeze()
x_test_norm = x_normaliser.transform(x_test)
y_test_norm = y_normaliser.transform(y_test).squeeze()

# Define function to produce the x sequences.

def create_x_sequences(x_array, num_sequences, tw):
    input_sequences = []
    for i in range(num_sequences):
        train_sequence = x_array[i:i+tw]
        input_sequences.append(train_sequence)
    input_sequences = np.stack(input_sequences, axis=0)
    return input_sequences


train_val_inputs = create_x_sequences(x_train_norm, len(y_train_norm), training_window)
test_inputs = create_x_sequences(x_test_norm, len(y_test_norm), training_window)
# print(train_val_inputs.shape, y_train_norm.shape)

# Split the training and validation sets
validation_size = 0.2
training_sequences, validation_sequences, training_targets, validation_targets = train_test_split(train_val_inputs, y_train_norm, test_size=validation_size, random_state=1)
print(f"\nTraining sequences {training_sequences.shape}\nValidation sequences {validation_sequences.shape}")
print(f"Training targets {training_targets.shape}\nValidation targets {validation_targets.shape}")
print(f"\nTest sequences {test_inputs.shape}\nTest targets {y_test_norm.shape}")

np.save(join(save_folder, "training_sequences.npy"), training_sequences)
np.save(join(save_folder, "validation_sequences.npy"), validation_sequences)
np.save(join(save_folder, f"training_targets_{age_category}.npy"), training_targets)
np.save(join(save_folder, f"validation_targets_{age_category}.npy"), validation_targets)

np.save(join(save_folder, "test_sequences.npy"), test_inputs)
np.save(join(save_folder, f"test_targets_{age_category}.npy"), y_test_norm)

print("\nSaved npy arrays.")
