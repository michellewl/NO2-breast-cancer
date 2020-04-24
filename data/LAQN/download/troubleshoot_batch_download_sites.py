import requests
import json
import pandas as pd
import os

problem_sites = ["HS5", "ME6"]

SpeciesCode = "NO2"
StartDate = "2002-01-01"
EndDate = "2018-01-01"

no2_df = pd.DataFrame()

for SiteCode in problem_sites:
    print(f"Site {SiteCode}")
    cur_df = pd.read_csv(f"http://api.erg.kcl.ac.uk/AirQuality/Data/SiteSpecies/"
                             f"SiteCode={SiteCode}/SpeciesCode={SpeciesCode}/StartDate={StartDate}/EndDate={EndDate}/csv")
    print("Downloaded.")
    cur_df.set_index("MeasurementDateGMT", drop=True, inplace=True)
    # print(cur_df.index)

    cur_df.index = cur_df.index+":00"
    print("Index fixed.")
    if no2_df.empty:
        no2_df = cur_df.copy()
    else:
        no2_df = no2_df.join(cur_df.copy(), how="left")

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
save_filepath = os.path.join(folder, f"NO2_2002-18_batch_troubleshooted.csv")
no2_df.to_csv(save_filepath)
print("Saved troubleshooted batch.")