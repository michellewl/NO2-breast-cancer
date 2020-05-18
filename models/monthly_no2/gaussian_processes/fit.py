import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import join, dirname, realpath, exists
import re
from sklearn.preprocessing import StandardScaler
import joblib
import sklearn.gaussian_process as gp
import matplotlib.pyplot as plt

### Code copied over from prev. work.

code_home_folder = "/home/mwlw3/Documents/Guided_team_challenge/building_resilience/"
data_folder = "data/processed_arrays/"
save_folder = "data/train_test_arrays/"
title = f"{code_home_folder}logs/training/daily_data/GP_log_{current_time()}"

print("\nFULL DATASET\n")
print("Reading dataset...")
files = glob.glob(f"{code_home_folder}{data_folder}full_dataframe_daily.csv")
data = pd.read_csv(files[0])
data["timestamp"] = pd.to_datetime(data.timestamp)
print("Processing dataset...")

y = data.meter_reading.to_numpy()
X = data[["mean_air_temp", "mean_wind_speed", "mean_RH", "min_air_temp", "max_air_temp"]].to_numpy()



print(f"X: {X.shape}\ny: {y.shape}")



print("Splitting into train and test sets...")
test_size = 0.15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("Applying normalisation...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train)
X_test = scaler.transform(X_test)
y_test = scaler.transform(y_test)


print("Saving train/test files...")

np.savetxt(f"{code_home_folder}{save_folder}X_train_GP.csv", X_train, delimiter=",")
np.savetxt(f"{code_home_folder}{save_folder}y_train_GP.csv", y_train, delimiter=",")
np.savetxt(f"{code_home_folder}{save_folder}X_test_GP.csv", X_test, delimiter=",")
np.savetxt(f"{code_home_folder}{save_folder}y_test_GP.csv", y_test, delimiter=",")

print("Saved.")
print("Fitting Gaussian process model...")

# set up covariance function
nu = 1
sigma = 1
lambbda = np.exp(-1)

kernel_1 = nu**2 * gp.kernels.RBF(length_scale=lambbda) + gp.kernels.WhiteKernel(noise_level=sigma)

# set up GP model
model_GP = gp.GaussianProcessRegressor(kernel=kernel_1)




write(title, f"{current_time()}\nKernel: {model_GP.kernel}")

model_GP.fit(X_train,y_train)
write(title, f"{current_time()}\nOptimised kernel: {model_GP.kernel}")
write(title, f"log marginal likelihood = {str(round(model_GP.log_marginal_likelihood(),4))}")


filename = f"GP_model.sav"
pickle.dump(model, open(filename, "wb"))

print("Saved model.")



# predict the mean function with 95% confidence error bars
mean_fn_plotx = np.linspace(-3, 3, 500)
mu, sigma_2 = model_GP.predict(mean_fn_plotx.reshape(-1,1), return_std=True)

fig, ax = plt.subplots(figsize=(10,6))

ax.fill_between(mean_fn_plotx, mu.squeeze() -2*sigma_2, mu.squeeze() +2*sigma_2, alpha=0.2)
ax.plot(mean_fn_plotx, mu.squeeze())
ax.scatter(x, y, color='black')
ax.set_title(model_GP.kernel_)
ax.annotate("log marginal likelihood = "+str(round(model_GP.log_marginal_likelihood(),4)), xy=(x.min(),y.max()*1.05), fontsize=12)
plt.show()