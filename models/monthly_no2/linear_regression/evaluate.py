import numpy as np
from os.path import join, dirname, realpath
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

ccgs = ["NHS Central London (Westminster)", "NHS Richmond"]
ccg = ccgs[0]
age_category = "all_ages"
test_year = 2017

# Load the arrays
load_folder = join(dirname(realpath(__file__)), ccg)
x_train, x_test = np.load(join(load_folder, "x_train.npy")), np.load(join(load_folder, "x_test.npy"))
y_train, y_test = np.load(join(load_folder, "y_train.npy")), np.load(join(load_folder, "y_test.npy"))

# Load normalisation
x_normaliser, y_normaliser = joblib.load(join(load_folder, "x_normaliser.sav")), \
                             joblib.load(join(load_folder, "y_normaliser.sav"))

# Normalise
x_train, x_test = x_normaliser.transform(x_train), x_normaliser.transform(x_test)
y_train, y_test = y_normaliser.transform(y_train), y_normaliser.transform(y_test)

# Load linear regression model
linear_regressor = joblib.load(join(load_folder, "linear_regressor.sav"))

# Evaluate the model
train_rsq_score = linear_regressor.score(x_train, y_train)
test_rsq_score = linear_regressor.score(x_test, y_test)
print(f"R squared on test set: {test_rsq_score}")

# Predict and un-normalise
y_train_predict = linear_regressor.predict(x_train)
y_train, y_train_predict = y_normaliser.inverse_transform(y_train), y_normaliser.inverse_transform(y_train_predict)

y_predict = linear_regressor.predict(x_test)
y_test, y_predict = y_normaliser.inverse_transform(y_test), y_normaliser.inverse_transform(y_predict)

# Plot prediction
train_dates = pd.date_range(f"2002-06", f"{test_year}-01", freq="M")
test_dates = pd.date_range(f"{test_year}-01", f"{test_year+1}-01", freq="M")

fig, axs = plt.subplots(2, 1, figsize=(15, 10))

axs[0].plot(train_dates, y_train, label="observed")
axs[0].plot(train_dates, y_train_predict, label="predicted")
axs[0].set_title(f"Training set (2002-06 to {test_year-1}-12)")
axs[0].annotate(f"R$^2$ = {train_rsq_score}", xy=(train_dates.min(), y_train.max()))
axs[1].plot(test_dates, y_test, label="observed")
axs[1].plot(test_dates, y_predict, label="prediction")
axs[1].set_title(f"Test set ({test_year})")
axs[1].annotate(f"R$^2$ = {test_rsq_score}", xy=(test_dates.min(), y_test.max()))

for ax in axs.flatten():
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Breast cancer cases ({age_category.replace( '_', ' ')}) per capita")

fig.suptitle(f"Linear regression for {ccg}")
fig.subplots_adjust(top=0.5)
plt.legend()
fig.tight_layout()
fig.savefig(join(load_folder, f"linear_regression_cases_{age_category}.png"), dpi=fig.dpi)
plt.show()
