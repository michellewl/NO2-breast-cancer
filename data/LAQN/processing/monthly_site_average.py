import os
import pandas as pd
import numpy as np

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # folder where data is saved
filename = [file for file in os.listdir(folder) if "averaged_ccgs.csv" in file][0]

no2_df = pd.read_csv(os.path.join(folder, filename))
no2_df.MeasurementDateGMT = pd.to_datetime(no2_df.MeasurementDateGMT)
no2_df.set_index("MeasurementDateGMT", inplace=True)
print(no2_df)

monthly_mean_df = no2_df.copy().resample("M").mean()
print(monthly_mean_df)

monthly_min_df = no2_df.copy().resample("M").min()
monthly_max_df = no2_df.copy().resample("M").max()
print(monthly_min_df, monthly_max_df)
