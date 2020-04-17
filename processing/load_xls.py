import xlrd
from glob import glob
import re

folder = "..\\data\\CCG_populations\\"
files = glob(f"{folder}*.xls*")

filenames = [file.replace(folder, "") for file in files]
print(filenames)

pattern = re.compile(r'\d\d\d\d')
years = [pattern.findall(filename)[0] for filename in filenames]
print(years)