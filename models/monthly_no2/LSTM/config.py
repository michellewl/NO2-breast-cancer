training_window = 3  # consider the last X months of NO2 for each breast cancer diagnosis month

aggregation = ["min", "max"]  # Choose from ["mean"], or ["min", "max"]. Make this False if not using.
quantile_step = False  # Make this False if not using.

ccgs = ["NHS Central London (Westminster)", "NHS Richmond"]
ccg = ccgs[0]
test_year = 2017

dates_as_inputs = False

age_category = "all_ages"

hidden_layer_size = 100
batch_size = 14
num_epochs = 5000
batches_per_print = False
epochs_per_print = 50
random_seed = 1

learning_rate = 0.001

model_epoch = "best"  # Choose "final" or "best" model.