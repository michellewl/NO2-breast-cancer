import requests
import json
import pandas as pd
from os.path import join, dirname, realpath, exists
from os import listdir, makedirs
import config

problem_sites = config.problem_sites

SpeciesCode = config.SpeciesCode
StartDate = config.StartDate
EndDate = config.EndDate

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

folder = join(dirname(dirname(realpath(__file__))), f"{StartDate}_{EndDate}")
save_filepath = join(folder, f"{SpeciesCode}_batch_troubleshooted.csv")
no2_df.to_csv(save_filepath)
print("Saved troubleshooted batch.")