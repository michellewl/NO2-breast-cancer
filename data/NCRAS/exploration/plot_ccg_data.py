import pandas as pd
import os
import re
import requests
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # folder where data is downloaded
files = [f for f in os.listdir(folder) if "ccgs.csv" in f.lower()] # lists filepaths (case-insensitive)
print(f"Filename: {files}")
filepath = os.path.join(folder, files[0])

cancer_df = pd.read_csv(filepath)
cancer_df.diagnosis_date = pd.to_datetime(cancer_df.diagnosis_date)

ccgs = cancer_df.ccg_name.unique()
print(len(ccgs))

timeseries, axs = plt.subplots(8, 4, figsize=(25, 20), sharex=True, sharey=True)
print(f"Plotting timeseries...")
timeseries.suptitle("Monthly breast cancer cases in London CCGs")
count = 0
for ax in axs.flat:
    ccg = ccgs[count]
    plot_df = cancer_df.loc[cancer_df.ccg_name==ccg]
    ax.plot(plot_df.diagnosis_date, plot_df.all_ages, alpha=0.7, c=f"C{count}")
    ax.set(xlabel="Time", ylabel="Number of cases (monthly)")
    ax.label_outer()
    ax.set_title(ccg)
    count += 1

timeseries.show()


histogram, axs = plt.subplots(8, 4, figsize=(25, 20), sharex=True, sharey=True)
print("Plotting histograms...")
histogram.suptitle("Monthly breast cancer cases in London CCGs")
count = 0
for ax in axs.flat:
    ccg = ccgs[count]
    plot_df = cancer_df.loc[cancer_df.ccg_name == ccg]
    ax.hist(plot_df.all_ages, alpha=0.7, color=f"C{count}")
    ax.set(xlabel="Monthly number of cases")
    ax.label_outer()
    ax.set_title(ccg)
    count += 1

histogram.show()

print("Saving plots...")

plot_filename = f"monthly_cases_ccg_timeseries.png"
timeseries.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), plot_filename))
print(f"\nSaved {plot_filename}")

plot_filename = f"monthly_cases_ccg_histograms.png"
histogram.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), plot_filename))
print(f"\nSaved {plot_filename}")

print("Completed save.")