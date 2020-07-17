import numpy as np

# Define function to produce the x sequences.

def create_x_sequences(x_array, num_sequences, tw):
    input_sequences = []
    for i in range(num_sequences):
        train_sequence = x_array[i:i + tw]
        input_sequences.append(train_sequence)
    input_sequences = np.stack(input_sequences, axis=0)
    return input_sequences

def mape_score(targets, predictions):
    zero_indices = np.where(targets == 0)
    targets_drop_zero = np.delete(targets, zero_indices)
    prediction_drop_zero = np.delete(predictions, zero_indices)
    mape = np.sum(np.abs(targets_drop_zero - prediction_drop_zero)/targets_drop_zero) * 100/len(targets_drop_zero)
    return mape