import torch.nn as nn


class TimeSeriesLSTM(nn.Module):
    def __init__(
        self, gap_prediction, window_size_y, input_size=12, hidden_size=64, num_layers=1, output_size=12
    ):
        super(TimeSeriesLSTM, self).__init__()

        self.gap_prediction = gap_prediction
        self.window_size_y = window_size_y

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, verbose=False):
        # Input x shape: (batch_size, sequence_length, input_size)

        # LSTM
        out, _ = self.lstm(x)  # out shape: (batch_size, sequence_length, hidden_size)

        if verbose:
            print("1:", out.shape)

        # Take the relevant steps from the LSTM output
        out = out[:, self.gap_prediction : self.gap_prediction + self.window_size_y, :]

        if verbose:
            print("2:", out.shape)

        # Linear layer to get the output
        out = self.linear(out)  # out shape: (batch_size, 4, output_size)

        if verbose:
            print("3:", out.shape)

        return out


# def test():
#     model = TimeSeriesLSTM()
#     total_params, trainable_params = count_parameters(model)
#     print(f"Total parameters: {total_params}")
#     print(f"Trainable parameters: {trainable_params}")

#     import torch.optim as optim

#     model = TimeSeriesLSTM().to("cuda")
#     model.train()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     input = T.randn(25, 107, 12, device="cuda")
#     out = model(input, verbose=True)
