import pandas as pd
from os.path import dirname, realpath, join, exists
from os import listdir, makedirs
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

folder = join(dirname(dirname(realpath(__file__))), "annual")  # folder where data is saved

annual_stat = "mean"
annual_df = pd.read_csv(join(folder, f"NO2_2002-18_ccgs_annual_{annual_stat}.csv"), index_col="MeasurementDateGMT")
annual_df.index = pd.to_datetime(annual_df.index)
# print(annual_df)

# # Lots of subplots
timeseries, axs = plt.subplots(8, 4, figsize=(25, 15), sharex=True, sharey=True)
print(f"\nPlotting {annual_stat} time series...")
timeseries.suptitle(rf"LAQN NO$_2$ annual {annual_stat}")
count = 0
for ax in axs.flat:
    column = annual_df.columns[count]
    ax.plot(annual_df.index, annual_df[column], alpha=0.7, c=f"C{count+1}")
    ax.set(xlabel="Time", ylabel=r"NO$_2$ ($\mu$g m$^{-3}$)")
    ax.label_outer()
    ax.set_title(column)
    count += 1

timeseries.show()

plot_filename = f"LAQN_NO2_timeseries_annual_{annual_stat}.png"
timeseries.savefig(join(dirname(realpath(__file__)), plot_filename))
print(f"Saved {plot_filename}")

#
# histogram, axs = plt.subplots(8, 4, figsize=(25, 20), sharex=True, sharey=True)
# print("Plotting histograms...")
# histogram.suptitle(r"LAQN observed NO$_2$ by CCG")
# count = 1
# for ax in axs.flat:
#     column = no2_df.columns[count]
#     ax.hist(no2_df[column], alpha=0.7, color=f"C{count}")
#     # ax.set(xlabel="Time", ylabel=r"NO$_2$ ($\mu$g m$^{-3}$)")
#     ax.label_outer()
#     ax.set_title(column)
#     count += 1
#
# histogram.show()
#
# print("Saving plots...")
#

#
# plot_filename = f"LAQN_NO2_histograms.png"
# histogram.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), plot_filename))
# print(f"\nSaved {plot_filename}")
#
# print("Completed save.")