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
no2_df.MeasurementDateGMT = pd.to_datetime(no2_df.MeasurementDateGMT)
no2_df.set_index("MeasurementDateGMT", inplace=True)
# print(f"{no2_df.shape[0]} time points.\n{no2_df.shape[1]} sites.")

meta_data = requests.get("http://api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName=London/Json")
meta_data_df = pd.DataFrame(meta_data.json()['Sites']['Site'])
meta_data_df = meta_data_df.loc[:,["@LocalAuthorityName", "@SiteName"]]
meta_data_df.columns = ["local_authority", "site_name"]
meta_data_df = meta_data_df.replace("- National Physical Laboratory, Teddington", "- National Physical Laboratory")
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

la_to_ccg_df = pd.DataFrame()

for site in odd_sites:
    for ccg in ccgs:
        if re.compile(site.split()[0]).findall(ccg):
            df = pd.DataFrame([(site, ccg)], columns=["local_authority", "ccg"])
            la_to_ccg_df = la_to_ccg_df.append(df, ignore_index=True)
#print(f"\nCheck local authorities have been mapped correctly:\n{la_to_ccg_df.tail(len(odd_sites))}")

site_mapping_df = meta_data_df.merge(la_to_ccg_df, on="local_authority")
print(f"\nSite mapping dataframe:\n{site_mapping_df.columns}")
# print(site_mapping_df.site_name)

# Try processing on one ccg first!
# Next try looping through all ccgs...

averaged_ccg_df = pd.DataFrame()

for ccg in ccgs:
    print(f"Processing {ccg}...")

    sites = site_mapping_df.loc[site_mapping_df.ccg == ccg, "site_name"].tolist()
    no2_ccg_df = no2_df.copy().reindex(columns=sites).dropna(axis="columns", how="all")
    # print(no2_ccg_df)

    # Step 1: Compute annual mean for each monitor for each year
    annual_mean_df = no2_ccg_df.resample("A").mean()
    # print(annual_mean_df)

    # Step 2: Subtract annual mean from daily measurements to obtain daily deviance for the monitor
    for year in annual_mean_df.index.year:
        for site in no2_ccg_df.columns:
            annual_mean = annual_mean_df.loc[annual_mean_df.index.year==year, site].tolist()*len(no2_ccg_df.loc[no2_ccg_df.index.year==year, site])
            no2_ccg_df.loc[no2_ccg_df.index.year==year, site] = no2_ccg_df.loc[no2_ccg_df.index.year==year, site] - annual_mean
    annual_mean_df[ccg] = annual_mean_df.mean(axis=1)
    # print(annual_mean_df)

    # Step 3: Standardise the daily deviance by dividing by standard deviation for the monitor
    sd_per_site = no2_ccg_df.copy().std(axis=0)
    sd_per_ccg = no2_ccg_df.copy().std(axis=1).std(axis=0)
    # print(sd_per_site)
    # print(sd_per_ccg)
    no2_ccg_df = no2_ccg_df/sd_per_site
    # print(no2_ccg_df)

    # Step 4: Average the daily standardised deviations to get an average across all monitors
    no2_ccg_df[ccg] = no2_ccg_df.mean(axis=1)
    # print(no2_ccg_df)

    # Step 5: Multiply the daily averaged standardised deviation
    # by the standard deviation across all monitor readings for the entire years (to un-standardise)
    no2_ccg_df[ccg] = no2_ccg_df[ccg] * sd_per_ccg
    # print(no2_ccg_df)

    # Step 6: Add the daily average deviance and annual average across all monitors to get a daily average reading
    for year in annual_mean_df.index.year:
        annual_mean = annual_mean_df.loc[annual_mean_df.index.year == year, ccg].tolist() * len(
            no2_ccg_df.loc[no2_ccg_df.index.year == year])

        no2_ccg_df.loc[no2_ccg_df.index.year==year, ccg] = \
            no2_ccg_df.loc[no2_ccg_df.index.year==year, ccg] + annual_mean
    no2_ccg_df = no2_ccg_df[[ccg]]

    if averaged_ccg_df.empty:
        averaged_ccg_df = no2_ccg_df.copy()
    else:
        averaged_ccg_df = averaged_ccg_df.join(no2_ccg_df.copy(), how="left")

print(averaged_ccg_df)
print(averaged_ccg_df.columns)

print("Saving full dataframe...")
save_filepath = os.path.join(folder, f"NO2_2002-18_averaged_ccgs.csv")
averaged_ccg_df.to_csv(save_filepath)
print("Completed save.")