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

sample_df = no2_df.sample(frac=0.1, axis="columns", random_state=random_seed)
sample_df.reset_index(inplace=True)
sample_df.MeasurementDateGMT = pd.to_datetime(sample_df.MeasurementDateGMT)

for column in sample_df.iloc[:, 1:].columns:
    print(f"Plotting {column}...")
    g = sns.relplot(x="MeasurementDateGMT", y=column, kind="line", data=sample_df, legend="full", height=5, aspect=3)

g.fig.suptitle(f"LAQN NO2")
plt.show()