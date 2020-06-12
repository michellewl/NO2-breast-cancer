from os.path import dirname, realpath, join, exists, basename
from os import makedirs
import requests
import zipfile as zp
import io

save_folder = join(dirname(realpath(__file__)), "London_GIS")
if not exists(save_folder):
    makedirs(save_folder)

urls = ["https://data.london.gov.uk/download/statistical-gis-boundary-files-london/9ba8c833-6370-4b11-abdc-314aa020d5e0/statistical-gis-boundaries-london.zip",
        "https://data.london.gov.uk/download/statistical-gis-boundary-files-london/b381c92b-9120-45c6-b97e-0f7adc567dd2/London-wards-2014.zip",
        "https://data.london.gov.uk/download/statistical-gis-boundary-files-london/08d31995-dd27-423c-a987-57fe8e952990/London-wards-2018.zip"
        ]

for url in urls:
    filename = basename(url)
    print(f"Downloading {filename}...")
    file_request = requests.get(url)
    zipfile = zp.ZipFile(io.BytesIO(file_request.content))
    print("Extracting...")
    zipfile.extractall(join(save_folder))

print("Complete.")
