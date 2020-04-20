from glob import glob
import re
import xlrd
import pandas as pd

def load_df_from_xlsheet(filename, sheet_name_pattern, verbose=False):
    """
    Function to load a pandas dataframe object from a Microsoft Excel spreadsheet file with specified sheet.

    :param filename: string object with full Excel file path
    :param sheet_name_pattern: string object with sheet name, or re compiler object for finding desired sheet in Excel file
    :param verbose: boolean value to choose whether to print sheet name
    :return: pandas dataframe object
    """
    if isinstance(sheet_name_pattern, str):
        return pd.read_excel(filename, sheet_name_pattern)
    else:
        workbook = xlrd.open_workbook(filename)
        sheets = workbook.sheet_names()

        for sheet in sheets:
            if sheet_name_pattern.findall(sheet):
                required_sheet = sheet

        if verbose:
            print(f"Sheet name: {required_sheet}")

        return pd.read_excel(filename, required_sheet)

def value_index(dataframe, column, value):
    """
    Obtains row index of a value in a dataframe, given the column name.
    :param dataframe: Pandas dataframe object.
    :param column: Name or index of column in dataframe.
    :param value: Value of interest.
    :return: Integer index.
    """
    if isinstance(column, str):
        return dataframe.loc[dataframe[column] == value].index[0]
    elif isinstance(column, int):
        return dataframe.loc[dataframe.iloc[:, column] == value].index[0]

def get_area_df(dataframe, area_col, area, sub_area=False):
    """
    Function to subset the full dataframe by area and/or sub-area.
    :param dataframe: Pandas dataframe object.
    :param area: String object of larger geographical area to choose. e.g. North of England
    :param sub_area: String object of sub-setted geographical area to choose. Optional. e.g. Lancashire
    :return: Pandas dataframe object of chosen geographical subset.
    """
    area_idx_start = value_index(dataframe, area_col, area)
    if isinstance(area_col, str):
        area_names = dataframe[area_col].copy().dropna().tolist()
    elif isinstance(area_col, int):
        area_names = dataframe.iloc[:, area_col].copy().dropna().tolist()
    area_idx_end = value_index(dataframe, area_col, area_names[area_names.index(area)+1])

    if sub_area:
        col_names = dataframe.copy().columns.tolist()

        if isinstance(area_col, str):
            col = col_names[col_names.index(area_col) + 1]
            names = dataframe[col].copy().dropna().tolist()
        elif isinstance(area_col, int):
            col = area_col + 1
            names = dataframe.iloc[:, col].copy().dropna().tolist()
        sub_idx_start = value_index(dataframe, col, sub_area)
        sub_idx_end = value_index(dataframe, col, names[names.index(sub_area) + 1])

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

def set_new_header(df, keyword):
    """
    Function to extract actual data table from Excel sheet with written blurb at the top.
    :param df: Pandas dataframe object of loaded Excel sheet.
    :param keyword: First column name.
    :return: Pandas dataframe object with header row starting with given keyword.
    """
    col_names = df.copy().columns.tolist()

    header_idx = df.index[df[col_names[0]] == keyword][0]
    new_header = df.iloc[header_idx].tolist()

    df = df.iloc[header_idx + 1:, ].reset_index(drop=True)
    df.columns = new_header
    return df

def get_sheet_names_from_xlfile(filename, pattern=False):
    """
    Function to print sheet names in an Excel file.
    :param filename: String object indicating path to Excel file.
    :param pattern: re compiler object (optional) to specify sheet name pattern.
    :return: List of sheet names as string objects.
    """

    workbook = xlrd.open_workbook(filename)
    sheets = workbook.sheet_names()
    if pattern:
        required_sheets = []
        for sheet in sheets:
            if pattern.findall(sheet):
                required_sheets.append(sheet)
        return required_sheets
    else:
        return sheets