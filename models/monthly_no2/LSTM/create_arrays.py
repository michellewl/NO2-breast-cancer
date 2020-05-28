import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import join, dirname, realpath, exists
import re
from sklearn.preprocessing import StandardScaler
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

print(f"x train: {x_train.shape}"
      f"\ny train: {y_train.shape}"
      f"\nx test: {x_test.shape}"
      f"\ny test: {y_test.shape}")

# Save the arrays
# if not exists(ccg):
#     makedirs(ccg)
# save_folder =

if quantile_step:
    aggregation = [str(len(aggregation)-1), "quantiles"]

# save_folder =
# if not exists(save_folder):
#     makedirs(save_folder)

save_folder = join(join(join(dirname(realpath(__file__)), ccg), "_".join(aggregation)), f"{training_window}_month_tw")
if not exists(save_folder):
    makedirs(save_folder)

np.save(join(save_folder, "x_train"), x_train)
np.save(join(save_folder, "x_test"), x_test)
np.save(join(save_folder, f"y_{age_category}_train"), y_train)
np.save(join(save_folder, f"y_{age_category}_test"), y_test)

# Normalise input and output training data
x_normaliser = StandardScaler().fit(x_train)
x_train = x_normaliser.transform(x_train)
y_normaliser = StandardScaler().fit(y_train)
y_train = y_normaliser.transform(y_train)

# Save normalisation to later apply to test sets
joblib.dump(x_normaliser, join(save_folder, "x_normaliser.sav"))
joblib.dump(y_normaliser, join(save_folder, f"y_{age_category}_normaliser.sav"))

print(f"Saved arrays to {save_folder}")