import requests
import json
import pandas as pd
import os

SiteCode = "HS5"

SpeciesCode = "NO2"
StartDate = "2002-01-01"
EndDate = "2018-01-01"

cur_df = pd.read_csv(f"http://api.erg.kcl.ac.uk/AirQuality/Data/SiteSpecies/"
                         f"SiteCode={SiteCode}/SpeciesCode={SpeciesCode}/StartDate={StartDate}/EndDate={EndDate}/csv")
cur_df.set_index("MeasurementDateGMT", drop=True, inplace=True)
print(cur_df.index)
#print(cur_df.columns)
#print(cur_df.head)

nan_count = cur_df.isnull().sum().sum()

print(f"\nNaN count: {nan_count}")