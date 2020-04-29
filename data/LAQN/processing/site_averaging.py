import numpy as np
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
# print(f"{no2_df.shape[0]} time points.\n{no2_df.shape[1]} sites.")

meta_data = requests.get("http://api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName=London/Json")
meta_data_df = pd.DataFrame(meta_data.json()['Sites']['Site'])
meta_data_df = meta_data_df.loc[:,["@LocalAuthorityName", "@SiteName"]]
meta_data_df.columns = ["local_authority", "site_name"]
local_auths = list(set(meta_data_df["local_authority"].tolist()))
local_auths.sort()
print(f"{len(local_auths)} local authorities.")

ccg_folder = os.path.join(os.path.dirname(folder), "CCG_populations")
ccg_filename = [file for file in os.listdir(ccg_folder) if "london_females_2002-18.csv" in file][0]
ccg_df = pd.read_csv(os.path.join(ccg_folder, ccg_filename))
ccgs = list(set(ccg_df["ccg"].tolist()))
# ccgs = [ccg.replace("NHS ", "") for ccg in list(set(ccg_df["ccg"].tolist()))]
print(f"{len(ccgs)} CCGs.")

odd_sites = [site for site in local_auths if site not in ccgs]
# matched_sites = [site for site in local_auths if site in ccgs]
# print(f"\n{len(odd_sites)} LAQN local authorities that don't match CCG names:\n{odd_sites}")

# la_to_ccg_df = pd.DataFrame(list(zip(matched_sites, matched_sites)), columns=["local_authority", "ccg"])
la_to_ccg_df = pd.DataFrame()

for site in odd_sites:
    for ccg in ccgs:
        if re.compile(site.split()[0]).findall(ccg):
            df = pd.DataFrame([(site, ccg)], columns=["local_authority", "ccg"])
            la_to_ccg_df = la_to_ccg_df.append(df, ignore_index=True)
#print(f"\nCheck local authorities have been mapped correctly:\n{la_to_ccg_df.tail(len(odd_sites))}")

site_mapping_df = meta_data_df.merge(la_to_ccg_df, on="local_authority")
print(f"\nSite mapping dataframe:\n{site_mapping_df.columns}")

# Try processing on one ccg first!

ccg = site_mapping_df.ccg[0]
#print(ccg)

sites = site_mapping_df.loc[site_mapping_df.ccg == ccg, "site_name"].tolist()
no2_ccg_df = no2_df.copy().reindex(columns=sites).dropna(axis="columns", how="all")
print(no2_ccg_df)
stats_ccg_df = pd.DataFrame()
