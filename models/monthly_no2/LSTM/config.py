SpeciesCode = "NO2"
laqn_start_date = "1997-01-01"
laqn_end_date = "2018-01-01"

training_window = 60  # consider the last X months of NO2 for each breast cancer diagnosis month

aggregation = False  # Choose from ["mean"], or ["min", "max"]. Make this False if not using.
quantile_step = 0.1  # Make this False if not using.

ccgs = ["clustered_ccgs"]  #["NHS Hammersmith and Fulham", "NHS Central London (Westminster)", "NHS Lambeth"]  # Cluster 3 of 4
# ["NHS Richmond"]
# ["all_ccgs"]
cluster_label = 3
n_clusters = 4
cluster_variables = "both_ncras_no2"

test_year = 2017

dates_as_inputs = False

age_category = "all_ages"

hidden_layer_size = 4
batch_size = 30
num_epochs = 5000
batches_per_print = False
epochs_per_print = 50
random_seed = 1

learning_rate = 0.001

model_epoch = "best"  # Choose "final" or "best" model.

compute_test_loss = True
noise_standard_deviation = 0.3  # Standard deviation of Gaussian noise
