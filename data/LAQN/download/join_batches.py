import pandas as pd
from os.path import join, dirname, realpath, exists
from os import listdir, makedirs
import re
import config

StartDate = config.StartDate
EndDate = config.EndDate
SpeciesCode = config.SpeciesCode


folder = join(dirname(dirname(realpath(__file__))), f"{StartDate}_{EndDate}")
batch_files = [file for file in listdir(folder)
               if re.compile(rf"{SpeciesCode}_batch\w+.csv").findall(file)]

no2_df = pd.DataFrame()

for batch_file in batch_files:
    batch = re.compile(r"batch\w+").findall(batch_file)[0]
    print(f"{batch}")
    filepath = join(folder, batch_file)
    batch_df = pd.read_csv(filepath, index_col="MeasurementDateGMT")
    batch_df.dropna(axis="columns", how="all", inplace=True)
    batch_df.columns = [column.replace(": Nitrogen Dioxide (ug/m3)", "") for column in batch_df.columns]
    batch_df.columns = [column.replace("=", "") for column in batch_df.columns]

    if no2_df.empty:
        no2_df = batch_df.copy()
    else:
        if bool(set(batch_df.columns.tolist()).intersection(no2_df.columns.tolist())):
            rename_dict = {}
            for x in list(set(batch_df.columns).intersection(no2_df.columns)):
                rename_dict.update({x: f"{x}_"})
                print(f"Renamed duplicated column:\n{rename_dict}")
            no2_df.rename(mapper=rename_dict, axis="columns", inplace=True)
        no2_df = no2_df.join(batch_df.copy(), how="left")

# print(no2_df.columns)
# print(no2_df)
print("Saving full dataframe.")
save_filepath = join(folder, f"NO2_all_sites.csv")
no2_df.to_csv(save_filepath)
print("Completed save.")
