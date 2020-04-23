import re
import pandas as pd
#import functions as fn
import os
import glob

folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "CCG_populations") # folder where data is saved
filename = [file for file in os.listdir(folder) if ".csv" in file][0]

df = pd.read_csv(os.path.join(folder, filename), index_col=["area_code", "ccg"])
all_ages_columns = [column for column in df.columns if re.compile(r"\d\d\d\d_all_ages").match(column)]
all_ages_df = df[all_ages_columns]
all_ages_df.columns = [int(col.replace("_all_ages","")) for col in all_ages_df.columns]

print(all_ages_df)