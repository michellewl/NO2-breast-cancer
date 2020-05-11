import pandas as pd
import os
import re
import requests
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # folder where data is saved
filename = [file for file in os.listdir(folder) if "averaged_ccgs.csv" in file][0]

no2_df = pd.read_csv(os.path.join(folder, filename))
no2_df.MeasurementDateGMT = pd.to_datetime(no2_df.MeasurementDateGMT)

# # One plot with all CCGs
# fig, axs = plt.subplots(figsize=(20, 8))
# fig.suptitle(r"LAQN observed NO$_2$ by CCG")
# for column in no2_df.iloc[:, 1:].columns:
#     print(f"Plotting {column}...")
#     ax[].plot(no2_df.MeasurementDateGMT, no2_df[column], label=column, alpha=0.3)
#
# # plt.figtext(0.2, 0.8, f"Sampling {int(sample_fraction*100)}% of all sites with random seed ({random_seed})")
# ax.set(xlabel="Time", ylabel=r"NO$_2$ ($\mu$g m$^{-3}$)")
# ax.legend(loc="upper right")
# plt.show()

# # Lots of subplots
timeseries, axs = plt.subplots(8, 4, figsize=(25, 15), sharex=True, sharey=True)
print(f"Plotting timeseries...")
timeseries.suptitle(r"LAQN observed NO$_2$ by CCG")
count = 1
for ax in axs.flat:
    column = no2_df.columns[count]
    ax.plot(no2_df.MeasurementDateGMT, no2_df[column], alpha=0.7, c=f"C{count}")
    ax.set(xlabel="Time", ylabel=r"NO$_2$ ($\mu$g m$^{-3}$)")
    ax.label_outer()
    ax.set_title(column)
    count += 1

timeseries.show()


histogram, axs = plt.subplots(8, 4, figsize=(25, 20), sharex=True, sharey=True)
print("Plotting histograms...")
histogram.suptitle(r"LAQN observed NO$_2$ by CCG")
count = 1
for ax in axs.flat:
    column = no2_df.columns[count]
    ax.hist(no2_df[column], alpha=0.7, color=f"C{count}")
    # ax.set(xlabel="Time", ylabel=r"NO$_2$ ($\mu$g m$^{-3}$)")
    ax.label_outer()
    ax.set_title(column)
    count += 1

histogram.show()

print("Saving plots...")

plot_filename = f"LAQN_NO2_timeseries.png"
timeseries.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), plot_filename))
print(f"\nSaved {plot_filename}")

plot_filename = f"LAQN_NO2_histograms.png"
histogram.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), plot_filename))
print(f"\nSaved {plot_filename}")

print("Completed save.")