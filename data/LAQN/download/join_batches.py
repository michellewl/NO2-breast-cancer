import pandas as pd
import os

num_batches = 10
#batch = 0

batches = range(0, num_batches)

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
no2_df = pd.DataFrame()

for batch in batches:
    print(f"Batch {batch}")
    filepath = os.path.join(folder, f"NO2_2002-18_batch{batch}.csv")
    batch_df = pd.read_csv(filepath, index_col="MeasurementDateGMT")
    batch_df.dropna(axis="columns", how="all", inplace=True)
    if no2_df.empty:
        no2_df = batch_df.copy()
    else:
        no2_df = no2_df.join(batch_df.copy(), how="left")

# print(no2_df.columns)
# print(no2_df)
print("Saving full dataframe.")
save_filepath = os.path.join(folder, f"NO2_2002-18_all_sites.csv")
no2_df.to_csv(save_filepath)
print("Completed save.")
