import re
import pandas as pd
import functions as fn
import os
import numpy as np

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # folder where data is downloaded
files = [f for f in os.listdir(folder) if "tumours.csv" in f.lower()] # lists filepaths (case-insensitive)
print(f"Filename: {files}")
filepath = os.path.join(folder, files[0])

cancer_df = pd.read_csv(filepath)
cancer_df.diagnosis_date = pd.to_datetime(cancer_df.diagnosis_date)

print(cancer_df.columns)

age_categories = [col for col in cancer_df.columns if "age_cat" in col]

ccg_df = pd.DataFrame()
for age_cat in age_categories:
    ccg_df[age_cat] = cancer_df.groupby(["ccg_code","ccg_name", "diagnosis_date"])[age_cat].sum()
ccg_df["all_ages"] = ccg_df.sum(axis=1)
print(ccg_df)

save_filename = "female_breast_cancer_london_2002-17_ccgs.csv"
ccg_df.to_csv(os.path.join(folder, save_filename), index=True)
print(f"Saved to {save_filename}")