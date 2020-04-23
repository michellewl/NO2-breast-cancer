import re
import pandas as pd
#import functions as fn
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import numpy as np

folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "CCG_populations") # folder where data is saved
filename = [file for file in os.listdir(folder) if ".csv" in file][0]

df = pd.read_csv(os.path.join(folder, filename), index_col=["area_code", "ccg"])

age_groupings = ["all_ages", "below_age_40", "age_40_to_70", "above_age_70"]

for group in age_groupings:

    grouped_ages_columns = [column for column in df.columns if re.compile(rf"\d\d\d\d_{group}").match(column)]
    grouped_ages_df = df[grouped_ages_columns]
    grouped_ages_df.columns = [int(col.replace(f"_{group}","")) for col in grouped_ages_df.columns]

    years = np.repeat(grouped_ages_df.columns.tolist(), len(df))
    plot_df = pd.DataFrame(years, columns=["year"])
    plot_df["ccg"] = df.index.levels[1].tolist()*len(grouped_ages_df.columns)
    plot_df["population"] = np.nan
    plot_df = plot_df.sort_values(["ccg", "year"])
    populations = [grouped_ages_df.loc[grouped_ages_df.index.levels[1] == ccg].values[0] for ccg in grouped_ages_df.index.levels[1]]
    plot_df["population"] = np.array(populations).flatten()

    g = sns.relplot(x="year", y="population", hue="ccg", kind="line", data=plot_df, legend=False, height=5, aspect=3)
    group_name = group.replace("_", " ")
    g.fig.suptitle(f"Female population ({group_name}) of London CCGs")
    plt.show()