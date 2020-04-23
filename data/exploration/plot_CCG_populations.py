import re
import pandas as pd
#import functions as fn
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "CCG_populations") # folder where data is saved
filename = [file for file in os.listdir(folder) if ".csv" in file][0]

df = pd.read_csv(os.path.join(folder, filename), index_col=["area_code", "ccg"])
all_ages_columns = [column for column in df.columns if re.compile(r"\d\d\d\d_all_ages").match(column)]
all_ages_df = df[all_ages_columns]
all_ages_df.columns = [int(col.replace("_all_ages","")) for col in all_ages_df.columns]

#print(all_ages_df)
years = np.repeat(all_ages_df.columns.tolist(), len(df))
plot_df = pd.DataFrame(years, columns=["year"])
plot_df["ccg"] = df.index.levels[1].tolist()*len(all_ages_df.columns)
plot_df["population"] = np.nan
plot_df = plot_df.sort_values(["ccg", "year"])

# first_ccg = all_ages_df.loc[all_ages_df.index.levels[1] == "NHS Barking & Dagenham"]
# print(first_ccg)

populations = [all_ages_df.loc[all_ages_df.index.levels[1] == ccg].values[0] for ccg in all_ages_df.index.levels[1]]

plot_df["population"] = np.array(populations).flatten()
#print(plot_df)

sns.set(style="darkgrid")
#fig = plt.figure(figsize=(20,5))
g = sns.relplot(x="year", y="population", hue="ccg", kind="line", data=plot_df, legend=False, height=5, aspect=3)
#plt.plot(plot_df.year, plot_df.population, label=plot_df.ccg)
# leg = g._legend
# leg.set_bbox_to_anchor([0.5, 0.5])  # coordinates of lower left of bounding box
# leg._loc = 2  # if required you can set the loc
g.fig.suptitle("Female population (all ages) of London CCGs")
plt.show()