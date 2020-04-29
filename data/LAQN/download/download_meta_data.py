import requests
import pandas as pd
import os

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

london_sites = requests.get("http://api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName=London/Json")
london_sites_df = pd.DataFrame(london_sites.json()['Sites']['Site'])
save_filepath = os.path.join(folder, f"site_meta_data.csv")
london_sites_df.to_csv(save_filepath)

species = requests.get("http://api.erg.kcl.ac.uk/AirQuality/Information/Species/Json")
species_df = pd.DataFrame(species.json()["AirQualitySpecies"]["Species"])
save_filepath = os.path.join(folder, f"species_meta_data.csv")
species_df.to_csv(save_filepath)

print(f"\nSaved meta data to csv.")
