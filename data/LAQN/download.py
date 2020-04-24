# Adapted from code provided by Geoff Ma.

import requests
import json
import pandas as pd

# API for LAQN data is "/Data/Site/Wide/SiteCode={SiteCode}/StartDate={StartDate}/EndDate={EndDate}/csv"
# This returns raw data based on 'SiteCode', 'StartDate', 'EndDate'.
# Default time period is 'hourly'.
# Data returned in CSV format

london_sites = requests.get("http://api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName=London/Json")

london_sites_df = pd.DataFrame(london_sites.json()['Sites']['Site'] )
#print(london_sites_df.columns)
all_site_codes = london_sites_df["@SiteCode"].tolist()
#print(all_site_codes)

# ### Example to pull data from a single site
# pandas is clever enough to do this all for you!
SiteCode = "BG1"
SpeciesCode = "NO2"
StartDate = "2002-01-01"
EndDate = "2018-01-01"
single_site = pd.read_csv(f"http://api.erg.kcl.ac.uk/AirQuality/Data/SiteSpecies/"
                          f"SiteCode={SiteCode}/SpeciesCode={SpeciesCode}/StartDate={StartDate}/EndDate={EndDate}/csv")
print(single_site)