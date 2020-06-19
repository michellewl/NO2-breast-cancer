SpeciesCode = "NO2"
laqn_start_date = "1997-01-01"
laqn_end_date = "2018-01-01"

# sklearn can't do 3D inputs so keep this as False for now.
training_window = False  # consider the last X months of NO2 for each breast cancer diagnosis month

aggregation = False  # Choose from ["mean"], or ["min", "max"]. Make this False if not using.
quantile_step = 0.1  # Make this False if not using.

ccgs = ["clustered_ccgs"]  # ["clustered_ccgs"]  # ["NHS Richmond"] # ["all_ccgs"]
cluster_label = 1
n_clusters = 2
cluster_variables = "both_ncras_no2"

test_year = 2017

age_category = "age_cat_40-69"  # "age_cat_>=70"  #"age_cat_<40"  #"age_cat_40-69"  #"all_ages"

