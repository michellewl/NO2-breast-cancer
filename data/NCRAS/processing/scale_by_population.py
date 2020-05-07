import re
import pandas as pd
import functions as fn
import os
import numpy as np

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # folder where data is saved
files = [f for f in os.listdir(folder) if "ccgs.csv" in f.lower()]  # lists filepaths (case-insensitive)
print(f"Filename: {files[0]}")
filepath = os.path.join(folder, files[0])

cancer_df = pd.read_csv(filepath)
cancer_df.date = pd.to_datetime(cancer_df.date)
cancer_df.set_index(["date", "ccg_code", "ccg_name"], inplace=True)
print(cancer_df.columns)

population_folder = os.path.join(os.path.dirname(folder), "CCG_populations")
population_filepath = os.path.join(population_folder, "london_female_pop_monthly_2002-06_2018-06.csv")
pop_df = pd.read_csv(population_filepath)
pop_df.date = pd.to_datetime(pop_df.date)
pop_df.set_index(["date", "ccg_code", "ccg_name"], inplace=True)
print(pop_df.columns)


scaled_df = cancer_df.copy().join(pop_df, lsuffix="_cases", rsuffix="_pop")
print(scaled_df.columns)
# print(scaled_df)

age_categories = cancer_df.columns.values
print(age_categories)
scaled_df.dropna(thresh=len(age_categories)+1, inplace=True)
print(scaled_df)

for age_category in age_categories:
    print(age_category)
    case_column = scaled_df[f"{age_category}_cases"]#.values.to_numpy()
    population_column = scaled_df[f"{age_category}_pop"]#.values.to_numpy()
    #print(case_column)
    scaled_column = (case_column / population_column)
    scaled_df[f"{age_category}"] = scaled_column
    scaled_df.drop(columns=[f"{age_category}_cases", f"{age_category}_pop"], inplace=True)
print(scaled_df)


save_filename = "female_breast_cancer_london_2002-17_age_grouped_ccgs_population_fraction.csv"
scaled_df.to_csv(os.path.join(folder, save_filename), index=True)
print(f"Saved to {save_filename}")