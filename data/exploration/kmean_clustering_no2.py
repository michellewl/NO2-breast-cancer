from sklearn.cluster import KMeans
import numpy as np
from os import listdir, makedirs
from os.path import join, dirname, realpath, exists
import re
import pandas as pd

# X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
# print(X.shape)
#
# kmeans = KMeans(n_clusters=2, random_state=1)
# kmeans.fit(X)
# print(f"Assigned clusters {kmeans.labels_}")
#
# X_sample = np.array([[0, 0], [12, 3]])
# print(f"Assign clusters to X sample\n{X_sample}\n{kmeans.predict(X_sample)}")
#
# print(f"Cluster centres {kmeans.cluster_centers_}")

cluster_data = "NO2"  # "NCRAS"
cluster_start_year = 2008
cluster_end_year = 2018

laqn_start_date = "1997-01-01"
laqn_end_date = "2018-01-01"

quantile_step = 0.25  # Make this False if not using.
aggregation = False  # ["min", "max"]
ccgs = ["all_ccgs"]
age_category = "all_ages"

if quantile_step:
    aggregation = [f"{int(method*100)}_quantile" for method in np.round(np.arange(0, 1+quantile_step, quantile_step), 2).tolist()]


no2_folder = join(dirname(dirname(realpath(__file__))), "LAQN", f"{laqn_start_date}_{laqn_end_date}", "monthly")
no2_filenames = [file for method in aggregation for file in listdir(no2_folder) if re.findall(f"ccgs_monthly_{method}.csv", file)]
print(no2_filenames)

ncras_folder = join(dirname(dirname(realpath(__file__))), "NCRAS")
ncras_filenames = [f for f in listdir(ncras_folder) if "ccgs_population_fraction.csv" in f][0]
ncras_df = pd.read_csv(join(ncras_folder, ncras_filenames)).set_index("date")
if ccgs == ["all_ccgs"]:
    ccgs = ncras_df["ccg_name"].unique().tolist()
print(len(ccgs))

cluster_ccgs = ccgs.copy()
ccg_arrays = []
for ccg in ccgs:
    arrays = []
    for file in no2_filenames:
        df = pd.read_csv(join(no2_folder, file)).set_index("MeasurementDateGMT")
        df.index = pd.to_datetime(df.index)
        df = df.loc[(df.index.year >= cluster_start_year) & (df.index.year < cluster_end_year), ccg].resample("A").mean()
        array = df.values.reshape(-1, 1)
        arrays.append(array)
    ccg_array = np.concatenate(arrays, axis=1).reshape(1, -1, len(aggregation))  # Annual average of monthly min/max for one CCG.
    if np.isnan(ccg_array).any():
        print(f"NaNs in {ccg}, skipped.")
        cluster_ccgs.remove(ccg)
    else:
        ccg_arrays.append(ccg_array)
no2_array = np.concatenate(ccg_arrays, axis=0)
print(no2_array.shape)

no2_array = np.mean(no2_array, axis=1)  # 10-year average of monthly min/max for each CCG.
print(no2_array.shape)
print(f"Retained {len(cluster_ccgs)} CCGs.")

print("\nk-means clustering\n")
kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(no2_array)
# print(f"Assigned clusters {kmeans.labels_}")

for i in range(len(cluster_ccgs)):
    print(f"{cluster_ccgs[i]} assigned to cluster {kmeans.labels_[i]}")

print(f"Cluster centres {kmeans.cluster_centers_}")
