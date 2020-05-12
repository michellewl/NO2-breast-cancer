import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

no2_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "LAQN")
no2_filenames = [f for f in os.listdir(no2_folder) if re.findall("ccgs_monthly_\w+.csv", f)]
no2_filenames.sort() # sorts files to max, mean, min order
print(no2_filenames)

ncras_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "NCRAS")
ncras_filename = [f for f in os.listdir(ncras_folder) if "ccgs_population_fraction.csv" in f][0]
print(ncras_filename)

ccgs = ["NHS Central London (Westminster)", "NHS Richmond"]

ncras_df = pd.read_csv(os.path.join(ncras_folder, ncras_filename)).set_index("ccg_name").loc[ccgs]
print(ncras_df)

no2_max_df = pd.read_csv(os.path.join(no2_folder, no2_filenames[0])).set_index("MeasurementDateGMT").loc[ncras_df.date.unique().tolist(), ccgs]
print(no2_max_df)

no2_mean_df = pd.read_csv(os.path.join(no2_folder, no2_filenames[1])).set_index("MeasurementDateGMT").loc[ncras_df.date.unique().tolist(), ccgs]
print(no2_mean_df)

no2_min_df = pd.read_csv(os.path.join(no2_folder, no2_filenames[2])).set_index("MeasurementDateGMT").loc[ncras_df.date.unique().tolist(), ccgs]
print(no2_min_df)



age_categories = [col for col in ncras_df.columns if "age" in col]
print(age_categories)

westminster_fig, axs = plt.subplots(len(age_categories), len(no2_filenames), figsize=(30, 20))

no2_colours = ["navy", "C0", "teal"]
ncras_colours = ["C1", "C3", "C5", "C6"]

for i in range(len(age_categories)):
    axs[i, 0].plot(no2_max_df.index, no2_max_df.loc[:, ccgs[0]], c=no2_colours[0], alpha=0.6)
    axs[i, 0].set_ylabel("no2_max", c=no2_colours[0])
    axs2 = axs[i, 0].twinx()
    axs2.plot(ncras_df.loc[ccgs[0], "date"], ncras_df.loc[ccgs[0], age_categories[i]], c=ncras_colours[i], alpha=0.7)
    axs2.set_ylabel(age_categories[i], c=ncras_colours[i])

    axs[i, 1].plot(no2_mean_df.index, no2_mean_df.loc[:, ccgs[0]], c=no2_colours[1], alpha=0.6)
    axs[i, 1].set_ylabel("no2_mean", c=no2_colours[1])
    axs2 = axs[i, 1].twinx()
    axs2.plot(ncras_df.loc[ccgs[0], "date"], ncras_df.loc[ccgs[0], age_categories[i]], c=ncras_colours[i], alpha=0.7)
    axs2.set_ylabel(age_categories[i], c=ncras_colours[i])

    axs[i, 2].plot(no2_mean_df.index, no2_mean_df.loc[:, ccgs[0]], c=no2_colours[2], alpha=0.6)
    axs[i, 2].set_ylabel("no2_mean", c=no2_colours[2])
    axs2 = axs[i, 2].twinx()
    axs2.plot(ncras_df.loc[ccgs[0], "date"], ncras_df.loc[ccgs[0], age_categories[i]], c=ncras_colours[i], alpha=0.7)
    axs2.set_ylabel(age_categories[i], c=ncras_colours[i])


plt.show()


