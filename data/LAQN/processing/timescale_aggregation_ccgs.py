from os.path import dirname, realpath, join, exists
from os import listdir, makedirs
import pandas as pd
import numpy as np
import config

StartDate = config.StartDate
EndDate = config.EndDate
SpeciesCode = config.SpeciesCode

folder = join(dirname(dirname(realpath(__file__))), f"{StartDate}_{EndDate}")  # folder where data is saved
filename = [file for file in listdir(folder) if "averaged_ccgs.csv" in file][0]

no2_df = pd.read_csv(join(folder, filename))
no2_df.MeasurementDateGMT = pd.to_datetime(no2_df.MeasurementDateGMT)
no2_df.set_index("MeasurementDateGMT", inplace=True)
print(f"{SpeciesCode} dataframe {no2_df.shape}")

# aggregation = ["mean", "min", "max"]
quantile_step = 0.25  # "False" if not in use.
if quantile_step:
    aggregation = np.round(np.arange(0, 1+quantile_step, quantile_step), 2).tolist()
print(aggregation)

timescale = "monthly"
timescale_code = "M"  # "M" for monthly, "A" for annual

save_folder = join(folder, timescale)
if not exists(save_folder):
    makedirs(save_folder)

for method in aggregation:
    if method == "mean":
        aggregated_df = no2_df.copy().resample(timescale_code).mean()
    elif method == "min":
        aggregated_df = no2_df.copy().resample(timescale_code).min()
    elif method == "max":
        aggregated_df = no2_df.copy().resample(timescale_code).min()
    elif isinstance(method, float):
        aggregated_df = no2_df.copy().resample(timescale_code).quantile(method)
        method = f"{int(method*100)}_quantile"
    aggregated_df.to_csv(join(save_folder, f"NO2_2002-18_ccgs_{timescale}_{method}.csv"), index=True)

# monthly_mean_df = no2_df.copy().resample("M").mean()
# print(monthly_mean_df)
#
# monthly_min_df = no2_df.copy().resample("M").min()
# monthly_max_df = no2_df.copy().resample("M").max()
# print(monthly_min_df, monthly_max_df)
#
# monthly_mean_df.to_csv(os.path.join(folder, "NO2_2002-18_ccgs_monthly_mean.csv"), index=True)
# monthly_min_df.to_csv(os.path.join(folder, "NO2_2002-18_ccgs_monthly_min.csv"), index=True)
# monthly_max_df.to_csv(os.path.join(folder, "NO2_2002-18_ccgs_monthly_max.csv"), index=True)