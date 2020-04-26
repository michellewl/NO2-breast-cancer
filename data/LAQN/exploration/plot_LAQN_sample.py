import pandas as pd
import os
import re
import requests
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # folder where data is saved
filename = [file for file in os.listdir(folder) if "all_sites.csv" in file][0]
no2_df = pd.read_csv(os.path.join(folder, filename))
no2_df.set_index("MeasurementDateGMT", inplace=True)

# site_strings = no2_df.columns[2].split()
# site_name = ' '.join([str(elem) for elem in site_strings[:site_strings.index("-")]])
# print(site_name)

# meta_data = requests.get("http://api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName=London/Json")
# meta_data_df = pd.DataFrame(meta_data.json()['Sites']['Site'])
# local_auths = meta_data_df["@LocalAuthorityName"].tolist()
#
# meta_data_df = meta_data_df.loc[:,["@LocalAuthorityName", "@SiteName"]]
# print(meta_data_df)

random_seed = 6
sample_fraction = 0.1

sample_df = no2_df.sample(frac=sample_fraction, axis="columns", random_state=random_seed)
sample_df.reset_index(inplace=True)
sample_df.MeasurementDateGMT = pd.to_datetime(sample_df.MeasurementDateGMT)

fig, ax = plt.subplots(figsize=(20, 8))
fig.suptitle(r"LAQN sample - NO$_2$")
for column in sample_df.iloc[:, 1:].columns:
    print(f"Plotting {column}...")
    ax.plot(sample_df.MeasurementDateGMT, sample_df[column], label=column, alpha=0.3)

plt.figtext(0.2, 0.8, f"Sampling {int(sample_fraction*100)}% of all sites with random seed ({random_seed})")
ax.set(xlabel="Time", ylabel=r"NO$_2$ ($\mu$g m$^{-3}$)")
ax.legend(loc="upper right")
# plt.show()

plot_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"LAQN_NO2_sample_seed{random_seed}.png")
fig.savefig(plot_filename)
print(f"\nSaved plot to:\n{plot_filename}")