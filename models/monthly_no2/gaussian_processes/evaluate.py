import numpy as np
import pandas as pd
from os.path import join, dirname, realpath, exists
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from functions import mape_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

kernel = "rbf"
aggregation = ["min", "max"]
# aggregation = ["mean"]
quantile_step = 0.1  # Make this False if not using.

ccgs = ["NHS Central London (Westminster)", "NHS Richmond"]
ccg = ccgs[1]
age_category = "all_ages"
test_year = 2017

if quantile_step:
    aggregation = [f"{int(method*100)}_quantile" for method in np.round(np.arange(0, 1+quantile_step, quantile_step), 2).tolist()]
print(aggregation)

# Load the arrays
if quantile_step:
    aggregation = [str(len(aggregation)-1), "quantiles"]
load_folder = join(join(dirname(realpath(__file__)), ccg), "_".join(aggregation))
x_train, x_test = np.load(join(load_folder, "x_train.npy")), np.load(join(load_folder, "x_test.npy"))
y_train, y_test = np.load(join(load_folder, f"y_{age_category}_train.npy")), np.load(join(load_folder, f"y_{age_category}_test.npy"))

# Load normalisation
x_normaliser, y_normaliser = joblib.load(join(load_folder, "x_normaliser.sav")), \
                             joblib.load(join(load_folder, f"y_{age_category}_normaliser.sav"))

# Normalise
x_train_norm, x_test_norm = x_normaliser.transform(x_train), x_normaliser.transform(x_test)
y_train_norm, y_test_norm = y_normaliser.transform(y_train), y_normaliser.transform(y_test)

# Load GP regression model
gp_regressor = joblib.load(join(load_folder, f"gp_regressor_{age_category}_{kernel}.sav"))


# Predict the mean function with 95% confidence error bars
mu_train, sigma_train = gp_regressor.predict(x_train_norm, return_std=True)
mu_test, sigma_test = gp_regressor.predict(x_test_norm, return_std=True)

# Calculate metrics
train_rsq = r2_score(y_train_norm, mu_train)
train_mse = mean_squared_error(y_train_norm, mu_train)
train_mape = mape_score(y_train_norm, mu_train)
print(f"\nTrain R sq {train_rsq}\nTrain MSE {train_mse}\nTrain MAPE {train_mape}")

test_rsq = r2_score(y_test_norm, mu_test)
test_mse = mean_squared_error(y_test_norm, mu_test)
test_mape = mape_score(y_test_norm, mu_test)
print(f"\nTest R sq {test_rsq}\nTest MSE {test_mse}\nTest MAPE {test_mape}")

# Plot time series

train_dates = pd.date_range(f"2002-06", f"{test_year}-01", freq="M")
test_dates = pd.date_range(f"{test_year}-01", f"{test_year+1}-01", freq="M")

fig, axs = plt.subplots(2, 1, figsize=(15, 10))

axs[0].fill_between(train_dates, mu_train.squeeze() - 2 * sigma_train, mu_train.squeeze() + 2 * sigma_train, alpha=0.2, label="std", color="C1")
axs[0].plot(train_dates, mu_train.squeeze(), label="mean function", color="C1")
axs[0].scatter(train_dates, y_train_norm, label="data (normalised)", color="C0")
axs[0].set_title(f"Training set (2002-06 to {test_year-1}-12)")
# axs[0].annotate(f"log marginal likelihood = {round(gp_regressor.log_marginal_likelihood(), 4)}",
axs[0].annotate(f"R$^2$ = {train_rsq}  MSE = {train_mse}  MAPE = {train_mape}",
                xy=(0.05, 0.92), xycoords="axes fraction", fontsize=12)


axs[1].fill_between(test_dates, mu_test.squeeze() - 2 * sigma_test, mu_test.squeeze() + 2 * sigma_test, alpha=0.2, label="std", color="C1")
axs[1].plot(test_dates, mu_test.squeeze(), label="mean function", color="C1")
axs[1].scatter(test_dates, y_test_norm, label="data (normalised)", color="C0")
axs[1].set_title(f"Test set ({test_year})")
# axs[0].annotate(f"Kernel: {gp_regressor.kernel_}",
axs[1].annotate(f"R$^2$ = {test_rsq}  MSE = {test_mse}  MAPE = {test_mape}",
                xy=(0.05, 0.92), xycoords="axes fraction", fontsize=12)

for ax in axs.flatten():
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Breast cancer cases ({age_category.replace( '_', ' ')}) per capita")

fig.suptitle(f"Gaussian process regression for {ccg}")

plt.figtext(0.1, 0.5, f"log marginal likelihood = {round(gp_regressor.log_marginal_likelihood(), 4)}",
            fontsize=12)
plt.figtext(0.1, 0.48, f"Kernel: {gp_regressor.kernel_}", fontsize=12)

plt.legend(loc=1)
fig.subplots_adjust(top=0.5)
fig.tight_layout(pad=2)

fig.savefig(join(load_folder, f"time_series_{age_category}_{kernel}.png"), dpi=fig.dpi)

# plt.show()

if not exists(join(load_folder, "metrics.txt")):
    metrics_log = open(join(load_folder, "metrics.txt"), "w")
else:
    metrics_log = open(join(load_folder, "metrics.txt"), "a")

metrics_log.write("METRICS LOG\n"
                  f"Age category: {age_category}\n"
                  f"Kernel: {kernel}\n"
                  f"Train R sq: {train_rsq}\n"
                  f"Train MSE: {train_mse}\n"
                  f"Train MAPE: {train_mape}\n\n"
                  f"Test R sq: {test_rsq}\n"
                  f"Test MSE: {test_mse}\n"
                  f"Test MAPE: {test_mape}\n\n\n")
metrics_log.close()