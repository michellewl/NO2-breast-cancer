import pandas as pd
import os

num_batches = 10
#batch = 0

batches = range(0, num_batches)

no2_df = pd.DataFrame()

for batch in batches:
    folder = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(folder, f"NO2_2002-18_batch{batch}.csv")
    if no2_df.empty:
        no2_df = pd.read_csv(filepath, index_col="MeasurementDateGMT")
    else:
        no2_df = no2_df.join(pd.read_csv(filepath, index_col="MeasurementDateGMT"), how="left")

# print(no2_df.columns)
# print(no2_df)

save_filepath = os.path.join(folder, f"NO2_2002-18_all_sites.csv")
no2_df.to_csv(save_filepath)