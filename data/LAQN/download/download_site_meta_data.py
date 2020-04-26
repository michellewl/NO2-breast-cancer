import requests
import pandas as pd
import os

london_sites = requests.get("http://api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName=London/Json")
london_sites_df = pd.DataFrame(london_sites.json()['Sites']['Site'])
london_sites_df.to_csv()

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
save_filepath = os.path.join(folder, f"meta_data.csv")
london_sites_df.to_csv(save_filepath)
print(f"\nSaved meta data to csv.")