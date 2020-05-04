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

population_folder = os.path.join(os.path.dirname(folder), "CCG_populations")
population_filepath = os.path.join(population_folder, "london_females_2002-18.csv")
pop_df = pd.read_csv(population_filepath)
print(pop_df.columns)

# # Align the CCG names
pop_ccgs = pop_df.ccg.unique()
pop_ccgs.sort()
ncras_ccgs = cancer_df.ccg_name.unique()
ncras_ccgs.sort()

odd_ccgs = [ccg for ccg in pop_ccgs if ccg not in ncras_ccgs]
# if odd_ccgs:
#     print(f"Mismatch in CCG names: {odd_ccgs}")

map_ccg_names = {}

for pop_ccg in odd_ccgs:
    for ncras_ccg in ncras_ccgs:
        if re.compile(f"{pop_ccg.split()[1]} ").findall(ncras_ccg):
            map_ccg_names.update({ncras_ccg: pop_ccg})

cancer_df = cancer_df.replace(map_ccg_names)

cancer_df["year"] = pd.DatetimeIndex(cancer_df.diagnosis_date).year

# age_categories = [col for col in cancer_df.columns if "age_cat" in col]
#
# ccg_df = pd.DataFrame()
# for age_cat in age_categories:
#     ccg_df[age_cat] = cancer_df.groupby(["ccg_code","ccg_name", "diagnosis_date"])[age_cat].sum()



# # Sum all the age groups
# ccg_df["all_ages"] = ccg_df.sum(axis=1)
# print(ccg_df)
#
# save_filename = "female_breast_cancer_london_2002-17_ccgs.csv"
# ccg_df.to_csv(os.path.join(folder, save_filename), index=True)
# print(f"Saved to {save_filename}")