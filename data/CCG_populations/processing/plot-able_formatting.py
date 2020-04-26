import re
import pandas as pd
#import functions as fn
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import numpy as np

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # folder where data is saved
filename = [file for file in os.listdir(folder) if file == "london_all_years.csv"][0]

df = pd.read_csv(os.path.join(folder, filename), index_col=["area_code", "ccg"])

age_groupings = ["all_ages", "below_age_40", "age_40_to_70", "above_age_70"]

full_df = pd.DataFrame()

for group in age_groupings:
    grouped_ages_df = df[[column for column in df.columns if re.compile(rf"\d\d\d\d_{group}").match(column)]]
    grouped_ages_df.columns = [int(col.replace(f"_{group}","")) for col in grouped_ages_df.columns]

    plot_df = pd.DataFrame(np.repeat(grouped_ages_df.columns.tolist(), len(df)), columns=["year"])
    plot_df["ccg"] = df.index.levels[1].tolist()*len(grouped_ages_df.columns)
    plot_df = plot_df.sort_values(["ccg", "year"])
    plot_df["population"] = np.array([grouped_ages_df.loc[grouped_ages_df.index.levels[1] == ccg].values[0]
                                      for ccg in grouped_ages_df.index.levels[1]]).flatten()
    plot_df["age_group"] = group
    # print(plot_df.columns)

    if full_df.empty:
        full_df = plot_df.copy()
    else:
        full_df = pd.concat([full_df, plot_df])

# print(full_df.columns)
print(full_df.head())

full_df.to_csv(os.path.join(folder, "full_encode_london_all_years.csv"), index=False)
print("Saved as .csv file.")