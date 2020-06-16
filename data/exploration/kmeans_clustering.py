from sklearn.cluster import KMeans
import numpy as np
from os import listdir
from os.path import join, dirname, realpath
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import adjustText as aT

variable = "both_ncras_no2"  # ncras or no2
cluster_start_year = 2013
cluster_end_year = 2018
number_of_clusters = 6

laqn_start_date = "1997-01-01"
laqn_end_date = "2018-01-01"

quantile_step = 0.1  # Make this False if not using.
aggregation = False  # ["min", "max"]
ccgs = ["all_ccgs"]
age_category = "all_ages"

if variable == "ncras":
    variable_name = "breast cancer"
elif variable == "no2":
    variable_name = "NO$_2$"
elif variable == "both_ncras_no2":
    variable_name = "breast cancer and NO$_2$"

if quantile_step:
    aggregation = [f"{int(method*100)}_quantile" for method in np.round(np.arange(0, 1+quantile_step, quantile_step), 2).tolist()]

# Get NO2 data filenames
no2_folder = join(dirname(dirname(realpath(__file__))), "LAQN", f"{laqn_start_date}_{laqn_end_date}", "monthly")
no2_filenames = [file for method in aggregation for file in listdir(no2_folder) if re.findall(f"ccgs_monthly_{method}.csv", file)]
print(f"\nNO2 filenames:\n{no2_filenames}")

# Get NCRAS data filename
ncras_folder = join(dirname(dirname(realpath(__file__))), "NCRAS")
ncras_filenames = [f for f in listdir(ncras_folder) if "ccgs_population_fraction.csv" in f]
print(f"NCRAS filenames:\n{ncras_filenames}")
ncras_df = pd.read_csv(join(ncras_folder, ncras_filenames[0])).set_index("date")
ncras_df.index = pd.to_datetime(ncras_df.index)

# Get list of all CCG names
if ccgs == ["all_ccgs"]:
    ccgs = ncras_df["ccg_name"].unique().tolist()
print(f"\n{len(ccgs)} CCGs found.")

# Calculate data on which to perform clustering
print(f"Variable for clustering: {variable_name}\n")
cluster_ccgs = ccgs.copy()
ccg_arrays = []
for ccg in ccgs:
    if variable == "no2":
        arrays = []
        for file in no2_filenames:
            df = pd.read_csv(join(no2_folder, file)).set_index("MeasurementDateGMT")
            df.index = pd.to_datetime(df.index)
            df = df.loc[(df.index.year >= cluster_start_year) & (df.index.year < cluster_end_year), ccg].resample("A").mean()
            array = df.values.reshape(-1, 1)
            arrays.append(array)
        ccg_array = np.concatenate(arrays, axis=1).reshape(1, -1, len(aggregation))  # Annual average of monthly X for one CCG.
    elif variable == "ncras":
        ccg_array = ncras_df.loc[(ncras_df.index.year >= cluster_start_year)
                                 & (ncras_df.index.year < cluster_end_year) & (ncras_df["ccg_name"] == ccg),
                                 age_category].resample("A").mean().values.reshape(1, -1)
    elif variable == "both_ncras_no2":
        arrays = []
        for file in no2_filenames:
            df = pd.read_csv(join(no2_folder, file)).set_index("MeasurementDateGMT")
            df.index = pd.to_datetime(df.index)
            df = df.loc[(df.index.year >= cluster_start_year) & (df.index.year < cluster_end_year), ccg].resample(
                "A").mean()
            array = df.values.reshape(-1, 1)
            arrays.append(array)
        no2_ccg_array = np.concatenate(arrays, axis=1).reshape(1, -1,
                                                           len(aggregation))  # Annual average of monthly X for one CCG.
        # print(no2_ccg_array.shape)
        ncras_ccg_array = ncras_df.loc[(ncras_df.index.year >= cluster_start_year)
                                 & (ncras_df.index.year < cluster_end_year) & (ncras_df["ccg_name"] == ccg),
                                 age_category].resample("A").mean().values.reshape(1, -1, 1)
        # print(ncras_ccg_array.shape)
        ccg_array = np.concatenate([no2_ccg_array, ncras_ccg_array], axis=2)
    if np.isnan(ccg_array).any():
        print(f"NaNs in {ccg}, skipped.")
        cluster_ccgs.remove(ccg)
    else:
        ccg_arrays.append(ccg_array)
annual_ccgs_array = np.concatenate(ccg_arrays, axis=0)
print(f"\nAnnual data {annual_ccgs_array.shape}")

cluster_array = np.mean(annual_ccgs_array, axis=1).reshape(len(cluster_ccgs), -1)  # 5-year average of monthly X for each CCG.
print(f"{cluster_end_year - cluster_start_year}-year mean data {cluster_array.shape}")
print(f"\nRetained {len(cluster_ccgs)} CCGs.")

# Perform clustering
print("\nK-MEANS CLUSTERING\n")

kmeans = KMeans(n_clusters=number_of_clusters, random_state=1)
kmeans.fit(cluster_array)

print(f"Cluster centres\n{kmeans.cluster_centers_}\n")

# Create a dataframe of cluster labels for each CCG
ccg_cluster_df = pd.DataFrame()
ccg_cluster_df["ccg"] = cluster_ccgs
ccg_cluster_df["cluster_label"] = kmeans.labels_
# print(cluster_df.shape)
ccg_cluster_df.to_csv(f"{variable}_{number_of_clusters}_clusters_{cluster_start_year}-{cluster_end_year}.csv")

# Load London map
load_folder = join(dirname(realpath(__file__)), "London_GIS", "statistical-gis-boundaries-london", "ESRI")
filename = "London_Borough_Excluding_MHW.shp"
map_df = gpd.read_file(join(load_folder, filename))
print(map_df.columns)

# Set up borough names for map plotting
map_labels_df = map_df.copy()
map_labels_df["rep"] = map_labels_df["geometry"].representative_point()
map_labels_df["centre"] = map_labels_df["geometry"].centroid
map_labels_df.set_geometry("centre", inplace=True)

# Create dataframe of cluster labels for each London borough - we need this to be able to plot the map
borough_cluster_df = pd.DataFrame()
# Map London boroughs to clustered CCGs to get clustered boroughs.
for borough in map_df["NAME"]:
    for ccg in cluster_ccgs:
        cluster_label = ccg_cluster_df.loc[ccg_cluster_df["ccg"] == ccg, "cluster_label"].values[0]
        ccg_names = ccg.replace("NHS ", "").replace("(", "").replace(")", "").replace(",", "")
        match = set(borough.split()).intersection(ccg_names.split())
        if (len(match) == 1 and "and" not in match and "London" not in match) or (len(match) > 1):
            df = pd.DataFrame([(borough, cluster_label)], columns=["borough", "cluster_label"])
            borough_cluster_df = borough_cluster_df.append(df, ignore_index=True)
print(borough_cluster_df.shape)

# Now prepare a dataframe for plotting the London map.
merge_df = map_df.set_index("NAME").join(borough_cluster_df.set_index("borough"))
merge_df["cluster_label"] = merge_df["cluster_label"].astype("Int64")
#merge_df.fillna(value={"cluster_label": 0}, inplace=True)
print(merge_df.shape)

# Plot the London map
font_size = 20

# Create figure and set colour scheme
fig, ax = plt.subplots(1, figsize=(20, 12))
cmap = ListedColormap(sns.color_palette("Paired").as_hex())

# Plot London boroughs with centre point markers and labels
merge_df.plot(column="cluster_label", categorical=True, linewidth=0.8, ax=ax, edgecolor="grey", cmap=cmap, legend=True,
              legend_kwds={"fontsize": font_size*0.8, "label": "Clusters"},
              missing_kwds={"color": "lightgrey", "edgecolor": "grey", "hatch": "///", "label": "Missing values"})
map_labels_df.plot(ax=ax, marker="o", color="black", markersize=font_size*0.4)
borough_text = []
for x, y, label in zip(map_labels_df.geometry.x, map_labels_df.geometry.y, map_labels_df["NAME"]):
    borough_text.append(plt.text(x, y, label, fontsize=font_size*0.6))

# Remove x and y axes from the plot, set the title and annotation
ax.axis("off")
ax.set_title(f"London {variable_name} k-means clustering", fontsize=font_size*1)
ax.annotate(f"Clustered on data from {cluster_start_year} to {cluster_end_year}", xy=(0.1, .08),
            xycoords="figure fraction", horizontalalignment="left", verticalalignment="top", fontsize=font_size*0.8, color="#555555")

# Adjust spacing for the borough labels so that text boxes don't overlap
aT.adjust_text(borough_text, expand_points=(1, 1), expand_text=(1, 1))

# Display the plot and save as a PNG file.
plt.show()

fig.savefig(f"{variable}_{number_of_clusters}_clusters_{cluster_start_year}-{cluster_end_year}.png", dpi=fig.dpi)
