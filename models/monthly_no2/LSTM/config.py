SpeciesCode = "NO2"
laqn_start_date = "1997-01-01"
laqn_end_date = "2018-01-01"

training_window = 24  # consider the last X months of NO2 for each breast cancer diagnosis month

aggregation = False  # Choose from ["mean"], or ["min", "max"]. Make this False if not using.
quantile_step = 0.1  # Make this False if not using.

ccgs = ["clustered_ccgs"]  # ["clustered_ccgs"]  # ["NHS Richmond"] # ["all_ccgs"]
cluster_label = 1
n_clusters = 2
cluster_variables = "both_ncras_no2"

age_category = "all_ages"  # "age_cat_>=70"  #"age_cat_<40"  #"age_cat_40-69"  #"all_ages"

noise_standard_deviation = 0.3  # Standard deviation of Gaussian noise (been using 0.3)

test_year = 2017

hidden_layer_size = 4
batch_size = 30
num_epochs = 10000
batches_per_print = False
epochs_per_print = 25
random_seed = 1

learning_rate = 0.001

model_epoch = "best"  # Choose "final" or "best" model.

compute_test_loss = False
include_plot_title = False
include_metrics = False