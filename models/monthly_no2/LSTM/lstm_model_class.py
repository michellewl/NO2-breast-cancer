import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=1):
        super().__init__()  # runs init for the Module parent class
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)  # LSTM layers
        self.linear = nn.Linear(hidden_layer_size, output_size)  # Linear layers
        # Hidden cell variable contains previous hidden state and previous cell state.
        # self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
        #                     torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        # Pass input sequence through the lstm layer, which outputs the layer output, hidden state and cell state
        # at the current time step.

        lstm_out, hidden_state_cell_state = self.lstm(input_seq)#, self.hidden_cell)
        # print(lstm_out.shape, hidden_state_cell_state[0].shape, hidden_state_cell_state[1].shape)
        # Pass the lstm output to the linear layer, which calculates the dot product between
        # the layer input and weight matrix.
        prediction = self.linear(lstm_out[:, -1, :])  # We want the most recent hidden state
        # print(prediction.shape)
        return prediction.squeeze()