from glob import glob
import re
import xlrd
import pandas as pd
import CCG_popns_functions as fn

folder = "..\\data\\CCG_populations\\"
filepaths = glob(f"{folder}*.xls*")

# Separately process files with different formatting.

years_2002_to_10 =[]
year_2011 = []
years_2012_to_18 = []
for file in filepaths:
    if "2002" in file:
        years_2002_to_10.append(file)
    elif "2011" in file:
        year_2011.append(file)
    else:
        years_2012_to_18.append(file)

# Dealing with 2011 (data has different formatting).
# df = fn.load_df_from_xlsheet(year_2011[0], re.compile(r"\wemale"))
# col_names = df.copy().columns.tolist()
#
# london_df = fn.get_area_df(df, "London", "London")
#
# ccg_col = col_names[col_names.index("Area Name")+2]
# london_ccg_df = london_df.loc[london_df[ccg_col].notna()].drop(columns=["Area Name", col_names[col_names.index("Area Name")+1]])
# london_ccg_df.rename(columns= {ccg_col:"CCG", "90+":90},inplace=True)
# london_ccg_df = fn.group_ages(london_ccg_df, 0, 40)
# london_ccg_df = fn.group_ages(london_ccg_df, 40, 71)
# london_ccg_df = fn.group_ages(london_ccg_df, 71, 91)

# Dealing with 2012 onwards (data files have same formatting).
df = fn.load_df_from_xlsheet(years_2012_to_18[0], re.compile(r"\wemale"))
df = fn.set_new_header(df, "Area Codes ")
col_names = df.copy().columns.tolist()
#print(col_names)
london_df = fn.get_area_df(df, 1, "London", "NHS England London")
print(london_df)

