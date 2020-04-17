from glob import glob
import re
import xlrd
import pandas as pd

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

def value_index(dataframe, column_name, value):
    """
    Obtains row index of a value in a dataframe, given the column name.
    :param dataframe: Pandas dataframe object.
    :param column_name: Name of column in dataframe.
    :param value: Value of interest.
    :return: Integer index.
    """
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

def group_ages(dataframe, start, end):
    """
    Function to aggregate age groups.
    :param dataframe: Pandas dataframe object.
    :param start: Start age (inclusive).
    :param end: End age (exclusive).
    :return: Pandas dataframe object with grouped ages.
    """
    age_cols = list(range(start, end))

    if start==0:
        dataframe[f"below_{end}_years"] = dataframe[age_cols].sum(axis=1)
    elif end == 91:
        dataframe[f"above_{start-1}_years"] = dataframe[age_cols].sum(axis=1)
    else:
        dataframe[f"{start}_to_{end-1}_years"] = dataframe[age_cols].sum(axis=1)
    dataframe = dataframe.drop(columns=age_cols)
    return dataframe