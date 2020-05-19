import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import join, dirname, realpath, exists
import re
from sklearn.preprocessing import StandardScaler
import joblib
import sklearn.gaussian_process as gp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

no2_folder = join(join(dirname(dirname(dirname(dirname(realpath(__file__))))), "data"), "LAQN")
no2_filenames = [f for f in listdir(no2_folder) if re.findall("ccgs_monthly_\w+.csv", f)]
no2_filenames.sort()  # sorts files to max, mean, min order
# print(no2_filenames)

ncras_folder = join(join(dirname(dirname(dirname(dirname(realpath(__file__))))), "data"), "NCRAS")
ncras_filename = [f for f in listdir(ncras_folder) if "ccgs_population_fraction.csv" in f][0]
# print(ncras_filename)

ccgs = ["NHS Central London (Westminster)", "NHS Richmond"]

ncras_df = pd.read_csv(join(ncras_folder, ncras_filename)).set_index("ccg_name").loc[ccgs]

# print(ncras_df)

no2_max_df = pd.read_csv(join(no2_folder, no2_filenames[0])).set_index("MeasurementDateGMT").loc[
    ncras_df.date.unique().tolist(), ccgs]
no2_max_df.index = pd.to_datetime(no2_max_df.index)
# print(no2_max_df)

no2_mean_df = pd.read_csv(join(no2_folder, no2_filenames[1])).set_index("MeasurementDateGMT").loc[
    ncras_df.date.unique().tolist(), ccgs]
no2_mean_df.index = pd.to_datetime(no2_mean_df.index)
# print(no2_mean_df)

no2_min_df = pd.read_csv(join(no2_folder, no2_filenames[2])).set_index("MeasurementDateGMT").loc[
    ncras_df.date.unique().tolist(), ccgs]
no2_min_df.index = pd.to_datetime(no2_min_df.index)
# print(no2_min_df)

ncras_df.reset_index(inplace=True)
ncras_df.set_index("date", inplace=True)
ncras_df.index = pd.to_datetime(ncras_df.index)

age_categories = [col for col in ncras_df.columns if "age" in col]

# One CCG, one age category

ccg = ccgs[1]
age_category = age_categories[-1]
print(f"{ccg}\n{age_category}")

# Get data arrays and split x and y into train and test (prediction) sets.
test_year = 2017

x_train = np.concatenate((no2_max_df.loc[no2_max_df.index.year != test_year, ccg].values.reshape(-1, 1),
                          no2_mean_df.loc[no2_mean_df.index.year != test_year, ccg].values.reshape(-1, 1)), axis=1)
x_test = np.concatenate((no2_max_df.loc[no2_max_df.index.year == test_year, ccg].values.reshape(-1, 1),
                         no2_mean_df.loc[no2_mean_df.index.year == test_year, ccg].values.reshape(-1, 1)), axis=1)

y_train = ncras_df.loc[(ncras_df.index.year != test_year) & (ncras_df.ccg_name == ccg), age_category] \
    .values.reshape(-1, 1)
y_test = ncras_df.loc[(ncras_df.index.year == test_year) & (ncras_df.ccg_name == ccg), age_category] \
    .values.reshape(-1, 1)

print(f"x train: {x_train.shape}"
      f"\ny train: {y_train.shape}"
      f"\nx test: {x_test.shape}"
      f"\ny test: {y_test.shape}")

# Save the arrays
if not exists(ccg):
    makedirs(ccg)
save_folder = join(dirname(realpath(__file__)), ccg)
# print(save_folder)

np.save(join(save_folder, "x_train"), x_train)
np.save(join(save_folder, "x_test"), x_test)
np.save(join(save_folder, "y_train"), y_train)
np.save(join(save_folder, "y_test"), y_test)

# Normalise input and output training data
x_normaliser = StandardScaler().fit(x_train)
x_train = x_normaliser.transform(x_train)
y_normaliser = StandardScaler().fit(y_train)
y_train = y_normaliser.transform(y_train)

# Save normalisation to later apply to test sets
joblib.dump(x_normaliser, join(save_folder, "x_normaliser.sav"))
joblib.dump(y_normaliser, join(save_folder, "y_normaliser.sav"))

# Fit the Gaussian process regression model
# Try using one dimensional input.
# Code copied over from prev. work.

print("Fitting Gaussian process model...")

# set up covariance function
nu = 1
sigma_init = 0.5
lamda = np.exp(-1)

kernel_init = nu ** 2 * gp.kernels.RBF(length_scale=lamda) + gp.kernels.WhiteKernel(noise_level=sigma_init)

# set up GP model
model_GP = gp.GaussianProcessRegressor(kernel=kernel_init)
model_GP.fit(x_train, y_train)

# Save the GP model
joblib.dump(model_GP, join(save_folder, "gp_regressor.sav"))
