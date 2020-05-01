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
fig, axs = plt.subplots(8, 4, figsize=(20, 10))

fig.suptitle(r"LAQN observed NO$_2$ by CCG")
count = 1
for ax in axs.flat:
    column = no2_df.columns[count]
    ax.plot(no2_df.MeasurementDateGMT, no2_df[column], alpha=0.3, c=f"C{count}")
    ax.set(xlabel="Time", ylabel=r"NO$_2$ ($\mu$g m$^{-3}$)")
    ax.label_outer()
    ax.set_title(column)
    count += 1

# plt.figtext(0.2, 0.8, f"")


plt.show()

# for column in no2_df.iloc[:, 1:].columns:
#     g = sns.relplot(x="MeasurementDateGMT", y=column, kind="line", data=no2_df, legend=False, height=5, aspect=3)
#     g.fig.suptitle(f"{column}")
#     plt.show()

# plot_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"LAQN_NO2_by_CCG.png")
# fig.savefig(plot_filename)
# print(f"\nSaved plot to:\n{plot_filename}")