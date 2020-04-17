from glob import glob
import re
import xlrd
import pandas as pd
import CCG_popns_functions as fn

folder = "..\\data\\CCG_populations\\"
filepaths = glob(f"{folder}*.xls*")

recent_years = []
for file in filepaths:
    if "2002" in file:
        continue
    else:
        recent_years.append(file)



df = fn.load_df_from_xlsheet(recent_years[0], re.compile(r"\wemale"))
col_names = df.copy().columns.tolist()

london_df = fn.get_area_df(df, "London", "London")

ccg_col = col_names[col_names.index("Area Name")+2]

london_ccg_df = london_df.loc[london_df[ccg_col].notna()].drop(columns=["Area Name", col_names[col_names.index("Area Name")+1]])
london_ccg_df.rename(columns= {ccg_col:"CCG", "90+":90},inplace=True)




london_ccg_df = fn.group_ages(london_ccg_df, 0, 40)
london_ccg_df = fn.group_ages(london_ccg_df, 40, 71)
london_ccg_df = fn.group_ages(london_ccg_df, 71, 91)

print(london_ccg_df.columns)