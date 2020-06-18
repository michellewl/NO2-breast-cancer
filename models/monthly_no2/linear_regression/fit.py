import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import join, dirname, realpath, exists
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import config


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
age_category = age_category.replace("<", "").replace(">=", "")


# Determine the appropriate monthly aggregation statistics for NO2
if quantile_step:
    aggregation = f"{int(1/quantile_step)}_quantiles"
else:
    aggregation = "_".join(config.aggregation)
print(aggregation)

# Define the loading folder for the experiment

if len(ccgs) > 1:
    load_folder = join(dirname(realpath(__file__)), "_".join(ccgs), aggregation)
elif ccgs == ["clustered_ccgs"]:
    label = f"cluster_{config.cluster_label}of{config.n_clusters}"
    load_folder = join(dirname(realpath(__file__)), ccgs[0], label, aggregation)
else:
    load_folder = join(dirname(realpath(__file__)), ccgs[0], aggregation)

load_folder = join(load_folder, input_description)


# Load numpy arrays

x_train = np.load(join(load_folder, "training_inputs.npy"))
y_train = np.load(join(load_folder, f"training_targets_{age_category}.npy"))

# Fit the linear model
lin_regressor = LinearRegression().fit(x_train, y_train)
r_sq = lin_regressor.score(x_train, y_train)
print(f"R squared on training set: {r_sq}")
# Save the linear model
joblib.dump(lin_regressor, join(load_folder, "linear_regressor.sav"))




