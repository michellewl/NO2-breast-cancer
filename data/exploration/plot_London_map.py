import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from os.path import join, dirname, realpath

# set the filepath and load in a shapefile
load_folder = join(dirname(realpath(__file__)), "London_GIS", "statistical-gis-boundaries-london", "ESRI")
filename = "London_Borough_Excluding_MHW.shp"

map_df = gpd.read_file(join(load_folder, filename))

map_df.plot()
plt.show()