from glob import glob
import re
import xlrd
import pandas as pd
import CCG_popns_functions as fn

folder = "..\\data\\CCG_populations\\"
filepaths = glob(f"{folder}*.xls*")

#year = "all"
year = "2002-10"
#year = "2011"
#year = "2012-18"

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
if year == "2011" or year == "all":

    df = fn.load_df_from_xlsheet(year_2011[0], re.compile(r"\wemale"))
    col_names = df.copy().columns.tolist()

    london_df = fn.get_area_df(df, "Area Name", "London", "London")

    ccg_col = col_names[col_names.index("Area Name")+2]
    london_ccg_df = london_df.loc[london_df[ccg_col].notna()].drop(columns=["Area Name", col_names[col_names.index("Area Name")+1]])
    london_ccg_df.rename(columns= {"Area Code":"area_code", "All Ages":"all_ages", ccg_col:"ccg", "90+":90},inplace=True)
    london_ccg_df = fn.group_ages(london_ccg_df, 0, 40)
    london_ccg_df = fn.group_ages(london_ccg_df, 40, 71)
    london_ccg_df = fn.group_ages(london_ccg_df, 71, 91)
    print(f"Filename: {year_2011[0]}\nNumber of CCGs: {len(london_ccg_df)}")
    #print(f"{london_ccg_df.head(2)}")
    #print(london_ccg_df.columns)

# Dealing with 2012 onwards (data files have same formatting).
if year=="2012-18" or year=="all":
    for i in range(len(years_2012_to_18)):

        df = fn.load_df_from_xlsheet(years_2012_to_18[i], re.compile(r"\wemale"))
        df = fn.set_new_header(df, "Area Codes ")
        col_names = df.copy().columns.tolist()
        london_df = fn.get_area_df(df, 1, "London", "NHS England London")

        col_names[1] = "Unknown1"
        col_names[2] = "Unknown2"
        london_df.columns = col_names

        ccg_col = "Area Names"
        london_ccg_df = london_df.loc[london_df[ccg_col].notna()].drop(columns=["Unknown1", "Unknown2"])
        london_ccg_df.rename(columns= {"Area Codes ":"area_code", "All Ages":"all_ages", ccg_col:"ccg", "90+":90},inplace=True)
        london_ccg_df = fn.group_ages(london_ccg_df, 0, 40)
        london_ccg_df = fn.group_ages(london_ccg_df, 40, 71)
        london_ccg_df = fn.group_ages(london_ccg_df, 71, 91)
        print(f"Filename: {years_2012_to_18[i]}\nNumber of CCGs: {len(london_ccg_df)}")
        #print(f"{london_ccg_df.head(2)}")
        #print(london_ccg_df.columns)

# Dealing with 2002-2010 (different formatting)
if year == "2002-10" or year == "all":
    print(f"Filename: {years_2002_to_10}")

    # Get London area codes
    df = fn.load_df_from_xlsheet(year_2011[0], re.compile(r"\wemale"))
    london_df = fn.get_area_df(df, "Area Name", "London", "London")
    area_codes = london_df["Area Code"].dropna().tolist()
    del area_codes[0]

    # Load 2002 data
    filename = years_2002_to_10[0]
    required_sheets = fn.get_sheet_names_from_xlfile(filename, re.compile(r"20\d\d"))
    required_sheets.remove("Mid-2011")

    for i in range(len(required_sheets)):
        df = fn.load_df_from_xlsheet(filename, required_sheets[i])

        male_cols = []
        for column in df.columns:
            if re.compile(r"m\w+").match(column):
                male_cols.append(column)
        df = df.drop(columns=male_cols).drop(columns="all_ages")

        london_ccg_df =  df[df['Area_Code'].isin(area_codes)].copy()

        rename_dict = {}
        for column in london_ccg_df.columns:
            if re.compile(r"f\d+$").findall(column):
                rename_dict.update({column:int(column.replace("f", ""))})
        rename_dict.update({"f90plus" : 90})
        london_ccg_df.rename(columns= rename_dict,inplace=True)
        london_ccg_df.rename(columns={"Area_Code":"area_code", "Area_Name":"ccg"}, inplace=True)

        london_ccg_df = fn.group_ages(london_ccg_df, 0, 40)
        london_ccg_df = fn.group_ages(london_ccg_df, 40, 71)
        london_ccg_df = fn.group_ages(london_ccg_df, 71, 91)
        london_ccg_df["all_ages"] = london_ccg_df.below_age_40 + london_ccg_df.age_40_to_70 + london_ccg_df.above_age_70

        print(f"Sheet: {required_sheets[i]}\nNumber of CCGs: {len(london_ccg_df)}")
        print(london_ccg_df.head(2))
        #print(london_ccg_df.columns)



