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
print(london_sites_df.head)