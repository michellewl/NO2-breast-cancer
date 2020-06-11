from sklearn.cluster import KMeans
import numpy as np
from os import listdir, makedirs
from os.path import join, dirname, realpath, exists
import re

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

laqn_start_date = "1997-01-01"
laqn_end_date = "2018-01-01"

quantile_step = 0.25  # Make this False if not using.
aggregation = False
ccgs = ["NHS Central London (Westminster)"]
age_category = "all_ages"

if quantile_step:
    aggregation = [f"{int(method*100)}_quantile" for method in np.round(np.arange(0, 1+quantile_step, quantile_step), 2).tolist()]

if cluster_data == "NO2":
    folder = join(join(join(dirname(dirname(realpath(__file__))), "LAQN"), f"{laqn_start_date}_{laqn_end_date}"), "monthly")
    filenames = [file for method in aggregation for file in listdir(folder) if re.findall(f"ccgs_monthly_{method}.csv", file)]
elif cluster_data == "NCRAS":
    folder = join(dirname(dirname(realpath(__file__))), "NCRAS")
    filenames = [f for f in listdir(folder) if "ccgs_population_fraction.csv" in f][0]

print(filenames)