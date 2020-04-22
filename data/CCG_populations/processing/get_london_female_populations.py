import re
import pandas as pd
import functions as fn
import os

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # folder where data is downloaded
filepaths = [f for f in os.listdir(folder) if '.xls' in f.lower()] # lists filepaths for all Excel files (case-insensitive)

process_year = "all"
#process_year = "2002-10"
#process_year = "2011"
#process_year = "2012-18"

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

full_df = pd.DataFrame()

# Dealing with 2002-2010 (different formatting)
if process_year == "2002-10" or process_year == "all":
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
        year = re.compile(r"\d\d\d\d").findall(required_sheets[i])[0]
        df = fn.load_df_from_xlsheet(filename, required_sheets[i])

        male_cols = []
        for column in df.columns:
            if re.compile(r"m\w+").match(column):
                male_cols.append(column)
        df = df.drop(columns=male_cols).drop(columns="all_ages")

        london_ccg_df = df[df['Area_Code'].isin(area_codes)].copy()

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
        london_ccg_df = fn.rename_age_cols_with_year(london_ccg_df, year)
        print(f"Sheet: {required_sheets[i]}\nNumber of CCGs: {len(london_ccg_df)}")
        #print(london_ccg_df.head(2))
        #print(london_ccg_df.columns)
        london_ccg_df.set_index(keys = ["area_code", "ccg"], inplace=True)

        if full_df.empty:
            full_df = london_ccg_df.copy()
        else:
            full_df = full_df.join(london_ccg_df.copy(), how="left")

# Dealing with 2011 (data has different formatting).
if process_year == "2011" or process_year == "all":
    filename = year_2011[0]
    year = re.compile(r"\d\d\d\d").findall(filename)[0]
    df = fn.load_df_from_xlsheet(filename, re.compile(r"\wemale"))
    col_names = df.copy().columns.tolist()

    london_df = fn.get_area_df(df, "Area Name", "London", "London")

    ccg_col = col_names[col_names.index("Area Name")+2]
    london_ccg_df = london_df.loc[london_df[ccg_col].notna()].drop(columns=["Area Name", col_names[col_names.index("Area Name")+1]])
    london_ccg_df.rename(columns= {"Area Code":"area_code", "All Ages":"all_ages", ccg_col:"ccg", "90+":90},inplace=True)
    london_ccg_df = fn.group_ages(london_ccg_df, 0, 40)
    london_ccg_df = fn.group_ages(london_ccg_df, 40, 71)
    london_ccg_df = fn.group_ages(london_ccg_df, 71, 91)

    london_ccg_df = fn.rename_age_cols_with_year(london_ccg_df, "2011")
    print(f"Filename: {filename}\nNumber of CCGs: {len(london_ccg_df)}")
    # print(f"{london_ccg_df.head(2)}")
    # print(london_ccg_df.columns)
    london_ccg_df.set_index(keys=["area_code", "ccg"], inplace=True)
    if full_df.empty:
        full_df = london_ccg_df.copy()
    else:
        full_df = full_df.join(london_ccg_df.copy(), how="left")

# Dealing with 2012 onwards (data files have same formatting).
if process_year=="2012-18" or process_year=="all":
    for i in range(len(years_2012_to_18)):
        filename = years_2012_to_18[i]
        year = re.compile(r"\d\d\d\d").findall(filename)[0]

        df = fn.load_df_from_xlsheet(filename, re.compile(r"\wemale"))
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
        london_ccg_df = fn.rename_age_cols_with_year(london_ccg_df, year)
        print(f"Filename: {filename}\nNumber of CCGs: {len(london_ccg_df)}")
        #print(f"{london_ccg_df.head(2)}")
        #print(london_ccg_df.columns)
        london_ccg_df.set_index(keys = ["area_code", "ccg"], inplace=True)
        if full_df.empty:
            full_df = london_ccg_df.copy()
        else:
            full_df = full_df.join(london_ccg_df.copy(), how="left")

print(full_df.columns)


full_df.to_csv(f"{folder}london_all_years.csv")
print("Saved as .csv file.")
