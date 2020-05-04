import re
import pandas as pd
import functions as fn
import os
import numpy as np

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # folder where data is downloaded
files = [f for f in os.listdir(folder) if '.xls' in f.lower()] # lists filepaths for all Excel files (case-insensitive)
print(files)

filepath = os.path.join(folder, files[0])
df = fn.load_df_from_xlsheet(filepath, re.compile(r"Data\w+"))
print(df)