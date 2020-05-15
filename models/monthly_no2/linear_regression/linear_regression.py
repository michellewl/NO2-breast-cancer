import numpy as np
import pandas as pd
import os
import re
from sklearn.linear_model import LinearRegression
import pickle

no2_folder = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))), "data"), "LAQN")
no2_filenames = [f for f in os.listdir(no2_folder) if re.findall("ccgs_monthly_\w+.csv", f)]
no2_filenames.sort() # sorts files to max, mean, min order
# print(no2_filenames)

ncras_folder = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))), "data"), "NCRAS")
ncras_filename = [f for f in os.listdir(ncras_folder) if "ccgs_population_fraction.csv" in f][0]
# print(ncras_filename)

ccgs = ["NHS Central London (Westminster)", "NHS Richmond"]

ncras_df = pd.read_csv(os.path.join(ncras_folder, ncras_filename)).set_index("ccg_name").loc[ccgs]

# print(ncras_df)

no2_max_df = pd.read_csv(os.path.join(no2_folder, no2_filenames[0])).set_index("MeasurementDateGMT").loc[ncras_df.date.unique().tolist(), ccgs]
no2_max_df.index = pd.to_datetime(no2_max_df.index)
# print(no2_max_df)

no2_mean_df = pd.read_csv(os.path.join(no2_folder, no2_filenames[1])).set_index("MeasurementDateGMT").loc[ncras_df.date.unique().tolist(), ccgs]
no2_mean_df.index = pd.to_datetime(no2_mean_df.index)
# print(no2_mean_df)

no2_min_df = pd.read_csv(os.path.join(no2_folder, no2_filenames[2])).set_index("MeasurementDateGMT").loc[ncras_df.date.unique().tolist(), ccgs]
no2_min_df.index = pd.to_datetime(no2_min_df.index)
# print(no2_min_df)

ncras_df.reset_index(inplace=True)
ncras_df.set_index("date", inplace=True)
ncras_df.index = pd.to_datetime(ncras_df.index)

age_categories = [col for col in ncras_df.columns if "age" in col]

# One CCG, one age category

ccg = ccgs[0]
age_category = age_categories[-1]
print(f"{ccg}\n{age_category}")

# Get data arrays and split x and y into train and test (prediction) sets.
test_year = 2017

x_train = np.concatenate((no2_max_df.loc[no2_max_df.index.year != test_year, ccg].values.reshape(-1, 1),
                          no2_mean_df.loc[no2_mean_df.index.year != test_year, ccg].values.reshape(-1, 1)), axis=1)
x_test = np.concatenate((no2_max_df.loc[no2_max_df.index.year == test_year, ccg].values.reshape(-1, 1),
                         no2_mean_df.loc[no2_mean_df.index.year == test_year, ccg].values.reshape(-1, 1)), axis=1)

y_train = ncras_df.loc[(ncras_df.index.year != test_year) & (ncras_df.ccg_name == ccg), age_category]\
    .values.reshape(-1, 1)
y_test = ncras_df.loc[(ncras_df.index.year == test_year) & (ncras_df.ccg_name == ccg), age_category]\
    .values.reshape(-1, 1)

print(f"x train: {x_train.shape}"
      f"\ny train: {y_train.shape}"
      f"\nx test: {x_test.shape}"
      f"\ny test: {y_test.shape}")

# Normalise input and output training data


# Save normalisation to later apply to test sets
# Fit the linear model
# Save the linear model




