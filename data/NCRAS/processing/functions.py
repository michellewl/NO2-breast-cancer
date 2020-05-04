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

def get_sheet_names(filename, pattern=False):
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
            else:
                print("No sheets found matching this name.")
                return None
        return required_sheets
    else:
        return sheets

def one_hot(df, columns):
    '''
    one-hot encode variables in a specified column
    ----------
    Parameters:
    df (pandas dataframe)
    columns (list of strings): columns to one hot encode
    -------
    Return:
     One pandas dataframe with the one hot encoded columns
    '''
    new_df = pd.concat((df, pd.get_dummies(df[columns])), axis=1)
    return new_df.drop(columns, axis=1)