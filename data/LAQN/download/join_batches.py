import pandas as pd
import os
import glob
import re

#num_batches = 10
#batch = 0

#batches = range(0, num_batches)

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
batch_files = [re.compile(r"NO2_2002-18_batch\w+.csv").findall(file)[0] for file in os.listdir(folder)
               if re.compile(r"NO2_2002-18_batch\w+.csv").findall(file)]

no2_df = pd.DataFrame()

for batch_file in batch_files:
    batch = re.compile(r"batch\w+").findall(batch_file)[0]
    print(f"{batch}")
    filepath = os.path.join(folder, batch_file)
    batch_df = pd.read_csv(filepath, index_col="MeasurementDateGMT")
    batch_df.dropna(axis="columns", how="all", inplace=True)
    batch_df.columns = [column.replace(": Nitrogen Dioxide (ug/m3)", "") for column in batch_df.columns]
    #print(batch_df.columns.tolist())
    if no2_df.empty:
        no2_df = batch_df.copy()
    else:
        if bool(set(batch_df.columns.tolist()).intersection(no2_df.columns.tolist())):
            # print(list(set(batch_df.columns).intersection(no2_df.columns)))
            rename_dict = {}
            for x in list(set(batch_df.columns).intersection(no2_df.columns)):
                rename_dict.update({x: f"{x}_"})
                print(f"Renamed duplicated column:\n{rename_dict}")
            no2_df.rename(mapper=rename_dict, axis="columns", inplace=True)
        no2_df = no2_df.join(batch_df.copy(), how="left")

#print(no2_df.columns)
# print(no2_df)
print("Saving full dataframe.")
save_filepath = os.path.join(folder, f"NO2_2002-18_all_sites.csv")
no2_df.to_csv(save_filepath)
print("Completed save.")
