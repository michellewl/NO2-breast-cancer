import pandas as pd
import os
import re
import requests
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # folder where data is saved
# filename = [file for file in os.listdir(folder) if "all_sites.csv" in file][0]
# no2_df = pd.read_csv(os.path.join(folder, filename))
# no2_df.set_index("MeasurementDateGMT", inplace=True)
# print(f"{no2_df.shape[0]} time points.\n{no2_df.shape[1]} sites.")

meta_data = requests.get("http://api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName=London/Json")
meta_data_df = pd.DataFrame(meta_data.json()['Sites']['Site'])
# meta_data_df = meta_data_df.loc[:,["@LocalAuthorityName", "@SiteName"]]
local_auths = list(set(meta_data_df["@LocalAuthorityName"].tolist()))
print(f"{len(local_auths)} local authorities.")

ccg_folder = os.path.join(os.path.dirname(folder), "CCG_populations")
ccg_filename = [file for file in os.listdir(ccg_folder) if "london_females_2002-18.csv" in file][0]
ccg_df = pd.read_csv(os.path.join(ccg_folder, ccg_filename))
ccg_list = list(set(ccg_df["ccg"].tolist()))
print(f"{len(ccg_list)} CCGs.")