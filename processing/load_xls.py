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
print(df)