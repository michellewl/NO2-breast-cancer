import numpy as np
from os.path import join, dirname, realpath, exists
from os import makedirs
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
sns.set(style="darkgrid")
import config

# # aggregation = ["mean", "min", "max"]
# # aggregation = ["mean", "max"]
# # aggregation = ["mean"]
# quantile_step = 0.25  # Make this False if not using.
# aggregation = [f"{int(method*100)}_quantile" for method in np.round(np.arange(0, 1+quantile_step, quantile_step), 2).tolist()]
# print(aggregation)
#
# ccgs = ["NHS Central London (Westminster)", "NHS Richmond"]
# ccg = ccgs[0]
# age_category = "all_ages"
# test_year = 2017


# Define these from the config file
training_window = config.training_window
quantile_step = config.quantile_step
ccgs = config.ccgs
test_year = config.test_year
age_category = config.age_category
print(f"{ccgs}\n{age_category}")
if training_window:
    input_description = f"{training_window}-month_tw"
else:
    input_description = "paired_in_out"
print(input_description)
age_category_rename = age_category.replace("<", "").replace(">=", "")  # Can't have < or > in filepaths

# Determine the appropriate monthly aggregation statistic for NO2
if quantile_step:
    aggregation = f"{int(1/quantile_step)}_quantiles"
else:
    aggregation = "_".join(config.aggregation)

# Define the loading folder for the experiment

if len(ccgs) > 1:
    load_folder = join(dirname(realpath(__file__)), "_".join(ccgs), aggregation)
elif ccgs == ["clustered_ccgs"]:
    label = f"cluster_{config.cluster_label}of{config.n_clusters}"
    load_folder = join(dirname(realpath(__file__)), ccgs[0], label, aggregation)
else:
    load_folder = join(dirname(realpath(__file__)), ccgs[0], aggregation)

load_folder = join(load_folder, input_description)


# Load the arrays
# if quantile_step:
#     aggregation = [str(len(aggregation)-1), "quantiles"]
# load_folder = join(join(dirname(realpath(__file__)), ccg), "_".join(aggregation))
x_train_norm, x_test_norm = np.load(join(load_folder, f"training_inputs.npy")), np.load(join(load_folder, "test_inputs.npy"))
y_train_norm, y_test_norm = np.load(join(load_folder, f"training_targets_{age_category_rename}.npy")), np.load(join(load_folder, f"test_targets_{age_category_rename}.npy"))

# Load normalisation
x_normaliser = joblib.load(join(load_folder, "x_normaliser.sav"))
y_normaliser = joblib.load(join(load_folder, f"y_{age_category_rename}_normaliser.sav"))
#
# # Normalise
# x_train_norm, x_test_norm = x_normaliser.transform(x_train), x_normaliser.transform(x_test)
# y_train_norm, y_test_norm = y_normaliser.transform(y_train), y_normaliser.transform(y_test)

# Load linear regression model
linear_regressor = joblib.load(join(load_folder, "linear_regressor.sav"))

# Predict and un-normalise
y_train_predict_norm = linear_regressor.predict(x_train_norm)
y_train_predict = y_normaliser.inverse_transform(y_train_predict_norm)
y_train = y_normaliser.inverse_transform(y_train_norm)

y_predict_norm = linear_regressor.predict(x_test_norm)
y_predict = y_normaliser.inverse_transform(y_predict_norm)
y_test = y_normaliser.inverse_transform(y_test_norm)


# Compute the performance metrics
train_rsq = r2_score(y_train, y_train_predict)
train_mse = mean_squared_error(y_train, y_train_predict)
test_rsq = r2_score(y_test, y_predict)
test_mse = mean_squared_error(y_test, y_predict)
print(f"Test R sq {test_rsq}\nTest MSE {test_mse}")

# Load train/test dates and CCGs
training_dates_ccgs = np.load(join(load_folder, f"training_dates.npy"), allow_pickle=True)
test_dates_ccgs = np.load(join(load_folder, f"test_dates_{age_category_rename}.npy"), allow_pickle=True)

training_df, test_df = pd.DataFrame(training_dates_ccgs, columns=["date", "ccg"]).set_index("date"), pd.DataFrame(test_dates_ccgs, columns=["date", "ccg"]).set_index("date")
training_df["target"], training_df["prediction"] = y_train, y_train_predict
test_df["target"], test_df["prediction"] = y_test, y_predict
training_df.index, test_df.index = pd.to_datetime(training_df.index), pd.to_datetime(test_df.index)
print(f"\nTraining dataframe {training_df.shape}, Test dataframe {test_df.shape}")
ccgs = test_df["ccg"].unique()
print(f"{len(ccgs)} CCGs in test set")

# Plot prediction
# train_dates = pd.date_range(f"2002-06", f"{test_year}-01", freq="M")
# test_dates = pd.date_range(f"{test_year}-01", f"{test_year+1}-01", freq="M")

save_folder = join(load_folder, "results_plots")
if not exists(save_folder):
    makedirs(save_folder)

print("\nPlotting...")

for ccg in ccgs:
    print(ccg)
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    axs[0].plot(training_df.loc[training_df["ccg"] == ccg].index, training_df.loc[training_df["ccg"] == ccg, "target"], label="observed")
    axs[0].plot(training_df.loc[training_df["ccg"] == ccg].index, training_df.loc[training_df["ccg"] == ccg, "prediction"], label="predicted")
    axs[0].set_title(f"Training set ({training_df.index.min()} to {training_df.index.max()})")
    # axs[0].set_title(f"Training set (2002-06 to {test_year-1}-12)")
    axs[0].annotate(f"R$^2$ = {train_rsq}  MSE = {train_mse}", xy=(0.05, 0.92), xycoords="axes fraction")
    axs[1].plot(test_df.loc[test_df["ccg"] == ccg].index, test_df.loc[test_df["ccg"] == ccg, "target"], label="observed")
    axs[1].plot(test_df.loc[test_df["ccg"] == ccg].index, test_df.loc[test_df["ccg"] == ccg, "prediction"], label="prediction")
    axs[1].set_title(f"Test set ({test_year})")
    axs[1].annotate(f"R$^2$ = {test_rsq}  MSE = {test_mse}", xy=(0.05, 0.92), xycoords="axes fraction")

    for ax in axs.flatten():
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Breast cancer cases ({age_category.replace( '_', ' ')}) per capita")

    fig.suptitle(f"Linear regression for {ccg}")

    plt.legend(loc=1)
    fig.subplots_adjust(top=0.5)
    fig.tight_layout()

    # Simplify the CCG name for saving
    ccg = ccg.replace("NHS ", "").replace(" ", "_")
    try:
        ccg = ccg[:ccg.index("_(")]
    except:
        pass


    fig.savefig(join(save_folder, f"{ccg}_time_series_{age_category_rename}.png"), dpi=fig.dpi)
    # plt.show()

    # At the end of the loop, close the figure
    plt.close()

    if len(aggregation) == 2 and not quantile_step:
        # Visualise linear regression model
        plt.clf()
        #scatter_fig, ax = plt.subplots(figsize=(15, 10))

        scatter_fig = plt.figure(figsize=(13, 10))
        ax = scatter_fig.add_subplot(111, projection='3d')

        x_plot = np.concatenate((np.arange(x_train_norm.min(), x_train_norm.max()).reshape(-1,1),
                                np.arange(x_train_norm.min(), x_train_norm.max()).reshape(-1,1)),
                                axis=1)
        # y_plot = linear_regressor.predict(x_plot)

        x_plot, y_plot = np.meshgrid(x_plot[:, 0], x_plot[:, 1])
        z_plot = linear_regressor.predict(pd.DataFrame({'x': x_plot.ravel(), 'y': y_plot.ravel()}))

        ax.plot_surface(x_plot, y_plot, z_plot.reshape(x_plot.shape), rstride=1, cstride=1, alpha=0.3, color="C1")
        ax.scatter(x_train_norm[:, 0], x_train_norm[:, 1], y_train_norm[:, 0], label="training data (normalised)", alpha=1)
        ax.set_xlabel(f"monthly {aggregation[0]} $NO_2$")
        ax.set_ylabel(f"monthly {aggregation[1]} $NO_2$")
        ax.set_zlabel("breast cancer cases per capita")

        scatter_fig.suptitle(f"Linear regression for {ccg}")

        plt.legend(loc=1)
        scatter_fig.tight_layout()

        scatter_fig.savefig(join(save_folder, f"{ccg}_scatter_plot_{age_category_rename}.png"), dpi=scatter_fig.dpi)
        plt.show()