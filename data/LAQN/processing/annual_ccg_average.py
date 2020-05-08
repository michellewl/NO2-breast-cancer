import os
import pandas as pd

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # folder where data is saved
filename = [file for file in os.listdir(folder) if "averaged_ccgs.csv" in file][0]

no2_df = pd.read_csv(os.path.join(folder, filename))
no2_df.MeasurementDateGMT = pd.to_datetime(no2_df.MeasurementDateGMT)
no2_df.set_index("MeasurementDateGMT", inplace=True)
print(no2_df)

annual_df = no2_df.copy().resample("A").mean()
print(annual_df)

annual_df.to_csv(os.path.join(folder, "NO2_2002-18_ccgs_annual_mean.csv"), index=True)
