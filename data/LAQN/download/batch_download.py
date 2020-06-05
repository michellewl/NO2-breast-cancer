import requests
import pandas as pd
from os.path import join, dirname, realpath, exists
from os import makedirs
import config

batch = config.batch
num_batches = config.num_batches

SpeciesCode = config.SpeciesCode
StartDate = config.StartDate
EndDate = config.EndDate

# Default time period is 'hourly'.
# Data returned in CSV format

london_sites = requests.get("http://api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName=London/Json")
london_sites_df = pd.DataFrame(london_sites.json()['Sites']['Site'])
all_site_codes = london_sites_df["@SiteCode"].tolist()
print(f"{len(all_site_codes)} sites in total.")

print(f"\nBatch {batch}.")
no2_df = pd.DataFrame()
problem_sites = []

start_batch = int(batch*len(all_site_codes)/num_batches)
end_batch = int((batch+1)*len(all_site_codes)/num_batches)
if batch == len(all_site_codes)-1:
    end_batch == -1

for SiteCode in all_site_codes[start_batch:end_batch]:
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
        problem_sites.append(SiteCode)

if problem_sites:
    print(f"Unable to join sites: {problem_sites}")

dates_folder = join(dirname(dirname(realpath(__file__))), f"{StartDate}_{EndDate}")

if not exists(dates_folder):
    makedirs(dates_folder)

save_filepath = join(dates_folder, f"{SpeciesCode}_batch{batch}.csv")
no2_df.to_csv(save_filepath)
print(f"\nSaved batch {batch} to csv.")