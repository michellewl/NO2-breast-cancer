import re
import pandas as pd
import functions as fn
import os
import numpy as np

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # folder where data is downloaded
files = [f for f in os.listdir(folder) if '.xls' in f.lower()] # lists filepaths for all Excel files (case-insensitive)
print(f"Filename: {files}")

filepath = os.path.join(folder, files[0])

sheets = fn.get_sheet_names(filepath)
print(f"Excel sheets: {sheets}")

df = fn.load_df_from_xlsheet(filepath, sheets[1])

df.loc[:, "CCG_NAME"] = [ccg.replace(" CCG", "")for ccg in df.CCG_NAME]
df = df.rename(columns={"BEHAVIOUR_CODE_DESC": "behaviour", "RANK_VAR": "rank"})
print(df.columns)

df = fn.one_hot(df, ["AGE_CAT", "BEHAVIOUR_CODE_DESC", "RANK_VAR"])
df.columns = [column.lower() for column in df.columns]
df.columns = [column.replace(" ", "_") for column in df.columns]

print(df.columns)
print(df)
