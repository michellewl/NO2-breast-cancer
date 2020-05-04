import os
import requests

save_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
url = "http://www.ncin.org.uk/view?rid=4159"
filename = "OpenDataRelease_BreastCa_Fem_Diagnoses_2002_2017_London.xlsx"

print(f"Downloading {filename}...")
ncras_file = requests.get(url)
file = open(os.path.join(save_folder, filename), 'wb')
file.write(ncras_file.content)
file.close()
print("Saved.")

