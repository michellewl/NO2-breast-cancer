import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import join, dirname, realpath, exists
import re
from sklearn.preprocessing import StandardScaler
import joblib
import sklearn.gaussian_process as gp

kernel = "rq"
# aggregation = ["min", "max"]
# aggregation = ["mean"]
quantile_step = False  # Make this False if not using.

ccgs = ["NHS Central London (Westminster)", "NHS Richmond"]
ccg = ccgs[0]
test_year = 2017

if quantile_step:
    aggregation = [f"{int(method*100)}_quantile" for method in np.round(np.arange(0, 1+quantile_step, quantile_step), 2).tolist()]
print(aggregation)

# One age category
age_category = "all_ages"
print(f"{ccg}\n{age_category}")

# Load the arrays
if quantile_step:
    aggregation = [str(len(aggregation)-1), "quantiles"]
load_folder = join(join(dirname(realpath(__file__)), ccg), "_".join(aggregation))
x_train, x_test = np.load(join(load_folder, "x_train.npy")), np.load(join(load_folder, "x_test.npy"))
y_train, y_test = np.load(join(load_folder, f"y_{age_category}_train.npy")), np.load(join(load_folder, f"y_{age_category}_test.npy"))

# Load normalisation
x_normaliser, y_normaliser = joblib.load(join(load_folder, "x_normaliser.sav")), \
                             joblib.load(join(load_folder, f"y_{age_category}_normaliser.sav"))


print(f"x train: {x_train.shape}"
      f"\ny train: {y_train.shape}"
      f"\nx test: {x_test.shape}"
      f"\ny test: {y_test.shape}")

# Normalise input and output training data
x_train = x_normaliser.transform(x_train)
y_train = y_normaliser.transform(y_train)

############################### Gaussian process regression ###############################

# Fit the Gaussian process regression model
# Try using one dimensional input.
# Code copied over from prev. work.

print("Fitting Gaussian process model...")

# set up covariance function
scale_factor = 1
noise = 0.6
length_scale = np.exp(-1)
scale_mixture = 1.5

kernel_rbf = scale_factor ** 2 * gp.kernels.RBF(length_scale=length_scale) + gp.kernels.WhiteKernel(noise_level=noise)
kernel_rq = scale_factor**2 * gp.kernels.RationalQuadratic(length_scale=length_scale, alpha=scale_mixture) \
            + gp.kernels.WhiteKernel(noise_level=noise)

if kernel == "rbf":
    kernel_init = kernel_rbf
elif kernel == "rq":
    kernel_init = kernel_rq

# set up GP model
model_GP = gp.GaussianProcessRegressor(kernel=kernel_init)
model_GP.fit(x_train, y_train)

# Save the GP model
save_folder = load_folder
joblib.dump(model_GP, join(save_folder, f"gp_regressor_{age_category}_{kernel}.sav"))
print(f"Model saved.")
