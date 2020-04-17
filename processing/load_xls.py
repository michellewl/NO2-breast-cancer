from glob import glob
import re
import xlrd
import pandas as pd

folder = "..\\data\\CCG_populations\\"
filepaths = glob(f"{folder}*.xls*")

# pattern = re.compile(r'\d\d\d\d')
# years = [pattern.findall(file)[0] for file in filepaths]
# print(years)

recent_years = []
for file in filepaths:
    if "2002" in file:
        continue
    else:
        recent_years.append(file)

def load_df_from_xlsheet(filename, sheet_name_pattern):
    """
    Function to load a pandas dataframe object from a Microsoft Excel spreadsheet file with specified sheet.

    :param filename: string object with full Excel file path
    :param sheet_name_pattern: re compiler object for finding desired sheet in Excel file
    :return: pandas dataframe object
    """
    workbook = xlrd.open_workbook(filename)
    sheets = workbook.sheet_names()

    for sheet in sheets:
        if sheet_name_pattern.findall(sheet):
            required_sheet = sheet

    return pd.read_excel(filename, required_sheet)

df = load_df_from_xlsheet(recent_years[0], re.compile(r"\wemale"))

def value_index(dataframe, column_name, value):
        return dataframe.loc[dataframe[column_name]==value].index[0]

def get_area_df(dataframe, area, sub_area=False):
    """
    Function to subset the full dataframe by area and/or sub-area.
    :param dataframe: Pandas dataframe object.
    :param area: String object of larger geographical area to choose. e.g. North of England
    :param sub_area: String object of sub-setted geographical area to choose. Optional. e.g. Lancashire
    :return: Pandas dataframe object of chosen geographical subset.
    """
    area_idx_start = value_index(dataframe, "Area Name", area)
    area_names = dataframe["Area Name"].copy().dropna().tolist()
    area_idx_end = value_index(dataframe, "Area Name", area_names[area_names.index(area)+1])

    if sub_area:
        col_names = dataframe.copy().columns.tolist()
        col = col_names[col_names.index("Area Name") + 1]
        names = dataframe[col].copy().dropna().tolist()
        sub_idx_start = value_index(dataframe, col, area)
        sub_idx_end = value_index(dataframe, col, names[names.index(area)+1])
    else:
        sub_idx_start = area_idx_start
        sub_idx_end = area_idx_end

    idx_start = max(area_idx_start, sub_idx_start)
    idx_end = min(area_idx_end, sub_idx_end)

    return dataframe.iloc[idx_start:idx_end]


# ldn_idx_start = value_index(df, "Area Name", "London")
# area_names = df["Area Name"].copy().dropna().tolist()
#
# ldn_idx_end = value_index(df, "Area Name", area_names[area_names.index("London")+1])
# # print(ldn_idx_start, ldn_idx_end)
# #
# # print(df.iloc[ldn_idx_start:ldn_idx_end])
#
col_names = df.copy().columns.tolist()
#
# sub_area_col = col_names[col_names.index("Area Name")+1]
# sub_area_names = df[sub_area_col].copy().dropna().tolist()
#
# sub_area_idx_start = value_index(df, sub_area_col, "London")
# sub_area_idx_end = value_index(df, sub_area_col, sub_area_names[sub_area_names.index("London")+1])
#
# print(ldn_idx_end, sub_area_idx_end)

london_df = get_area_df(df, "London", "London")

ccg_col = col_names[col_names.index("Area Name")+2]

london_ccg_df = london_df.loc[london_df[ccg_col].notna()].drop(columns=["Area Name", col_names[col_names.index("Area Name")+1]])
london_ccg_df.rename(columns= {ccg_col:"CCG"},inplace=True)
print(london_ccg_df.columns)