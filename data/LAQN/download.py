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
london_sites_df = pd.DataFrame(london_sites.json()['Sites']['Site'])
all_site_codes = london_sites_df["@SiteCode"].tolist()
del london_sites
print(f"{len(all_site_codes)} sites in total.")

SpeciesCode = "NO2"
StartDate = "2002-01-01"
EndDate = "2018-01-01"

for batch in range(0, 10):
    print(f"\nBatch {batch}.")
    no2_df = pd.DataFrame()
    problem_sites = []

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
            problem_sites.append()

    if problem_sites:
        print(f"Unable to join sites: {problem_sites}")

    folder = os.path.dirname(os.path.realpath(__file__))
    save_filepath = os.path.join(folder, f"NO2_2002-18_batch{batch}.csv")
    print(f"Saved batch {batch} to csv.")