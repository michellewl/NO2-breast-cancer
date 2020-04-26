import pandas as pd
import os
import seaborn as sns
sns.set(style="darkgrid")

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # folder where data is saved
filename = [file for file in os.listdir(folder) if file == "london_all_years.csv"][0]
