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
            else:
                print("Sheet not found.")
                return None

        if verbose:
            print(f"Sheet name: {required_sheet}")

        return pd.read_excel(filename, required_sheet)