# Adapted from code provided by Geoff Ma.

import requests
import json
import pandas as pd
import os

# API for LAQN data is "/Data/Site/Wide/SiteCode={SiteCode}/StartDate={StartDate}/EndDate={EndDate}/csv"
# This returns raw data based on 'SiteCode', 'StartDate', 'EndDate'.
# Default time period is 'hourly'.
# Data returned in CSV format

london_sites = requests.get("http://api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName=London/Json")

london_sites_df = pd.DataFrame(london_sites.json()['Sites']['Site'] )
#print(london_sites_df.columns)
all_site_codes = london_sites_df["@SiteCode"].tolist()
print(f"{len(all_site_codes)} sites in total.")
#print(all_site_codes)

# ### Example to pull data from a single site
# pandas is clever enough to do this all for you!
# SiteCode = "BG1"
SpeciesCode = "NO2"
StartDate = "2002-01-01"
EndDate = "2018-01-01"
# single_site = pd.read_csv(f"http://api.erg.kcl.ac.uk/AirQuality/Data/SiteSpecies/"
#                           f"SiteCode={SiteCode}/SpeciesCode={SpeciesCode}/StartDate={StartDate}/EndDate={EndDate}/csv")
# print(single_site)

no2_df = pd.DataFrame()

for SiteCode in all_site_codes:
    print(f"\nWorking on site {SiteCode}. ({all_site_codes.index(SiteCode)} of {len(all_site_codes)})")
    cur_df = pd.read_csv(f"http://api.erg.kcl.ac.uk/AirQuality/Data/SiteSpecies/"
                         f"SiteCode={SiteCode}/SpeciesCode={SpeciesCode}/StartDate={StartDate}/EndDate={EndDate}/csv")
    print(f"Downloaded.")
    cur_df.set_index("MeasurementDateGMT", drop=True, inplace=True)

    try:
        no2_df = no2_df.join(cur_df, how="outer")
        print(f"Joined.")
    except KeyboardInterrupt:
        break
    except:
        print(f"Unable to join site {SiteCode} to dataframe.")

folder = os.path.dirname(os.path.realpath(__file__))
save_filepath = os.path.join(folder, "NO2_2002-18.csv")
print("Saved to csv.")