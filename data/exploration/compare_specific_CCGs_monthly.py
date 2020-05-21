import pandas as pd
from os.path import dirname, realpath, join, exists
from os import listdir, makedirs
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns
sns.set(style="darkgrid")

no2_folder = join(join(dirname(dirname(realpath(__file__))), "LAQN"), "monthly")
no2_filenames = [f for f in listdir(no2_folder) if re.findall("ccgs_monthly_\w+.csv", f)]
no2_filenames.sort() # sorts files to max, mean, min order
# print(no2_filenames)

ncras_folder = join(dirname(dirname(realpath(__file__))), "NCRAS")
ncras_filename = [f for f in listdir(ncras_folder) if "ccgs_population_fraction.csv" in f][0]
# print(ncras_filename)

ccgs = ["NHS Central London (Westminster)", "NHS Richmond"]

ncras_df = pd.read_csv(join(ncras_folder, ncras_filename)).set_index("ccg_name").loc[ccgs]

# print(ncras_df)

no2_max_df = pd.read_csv(join(no2_folder, no2_filenames[0])).set_index("MeasurementDateGMT").loc[ncras_df.date.unique().tolist(), ccgs]
no2_max_df.index = pd.to_datetime(no2_max_df.index)
# print(no2_max_df)

no2_mean_df = pd.read_csv(join(no2_folder, no2_filenames[1])).set_index("MeasurementDateGMT").loc[ncras_df.date.unique().tolist(), ccgs]
no2_mean_df.index = pd.to_datetime(no2_mean_df.index)
# print(no2_mean_df)

no2_min_df = pd.read_csv(join(no2_folder, no2_filenames[2])).set_index("MeasurementDateGMT").loc[ncras_df.date.unique().tolist(), ccgs]
no2_min_df.index = pd.to_datetime(no2_min_df.index)
# print(no2_min_df)

ncras_df.date = pd.to_datetime(ncras_df.date)

age_categories = [col for col in ncras_df.columns if "age" in col]
# print(age_categories)

print("Plotting...")

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

no2_colours = ["navy", "C0", "teal"]
ncras_colours = ["C1", "C3", "C5", "purple"]

for ccg in ccgs:
    print(f"... {ccg}")
    fig, axs = plt.subplots(len(age_categories), len(no2_filenames), figsize=(35, 20))

    for i in range(len(age_categories)):
        axs[i, 0].plot(no2_max_df.index, no2_max_df.loc[:, ccg], c=no2_colours[0], alpha=0.6)
        axs[i, 0].set_ylabel("no2_max", c=no2_colours[0])
        axs2 = axs[i, 0].twinx()
        axs2.plot(ncras_df.loc[ccg, "date"], ncras_df.loc[ccg, age_categories[i]], c=ncras_colours[i], alpha=0.7)
        axs2.set_ylabel(age_categories[i], c=ncras_colours[i])
        axs2.grid(False)
    
        axs[i, 1].plot(no2_mean_df.index, no2_mean_df.loc[:, ccg], c=no2_colours[1], alpha=0.6)
        axs[i, 1].set_ylabel("no2_mean", c=no2_colours[1])
        axs2 = axs[i, 1].twinx()
        axs2.plot(ncras_df.loc[ccg, "date"], ncras_df.loc[ccg, age_categories[i]], c=ncras_colours[i], alpha=0.7)
        axs2.set_ylabel(age_categories[i], c=ncras_colours[i])
        axs2.grid(False)
    
        axs[i, 2].plot(no2_min_df.index, no2_min_df.loc[:, ccg], c=no2_colours[2], alpha=0.6)
        axs[i, 2].set_ylabel("no2_min", c=no2_colours[2])
        axs2 = axs[i, 2].twinx()
        axs2.plot(ncras_df.loc[ccg, "date"], ncras_df.loc[ccg, age_categories[i]], c=ncras_colours[i], alpha=0.7)
        axs2.set_ylabel(age_categories[i], c=ncras_colours[i])
        axs2.grid(False)
    
    # format the ticks
    for ax in axs.flat:
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)

    fig.suptitle(f"{ccg}")
    
    plt.show()
    
    save_name = ccg.replace(" ", "_")
    fig.savefig(f"{save_name}.png")
