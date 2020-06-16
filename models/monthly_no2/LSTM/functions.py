import numpy as np

# Define function to produce the x sequences.

def create_x_sequences(x_array, num_sequences, tw):
    input_sequences = []
    for i in range(num_sequences):
        train_sequence = x_array[i:i + tw]
        input_sequences.append(train_sequence)
    input_sequences = np.stack(input_sequences, axis=0)
    return input_sequences