import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        gap_prediction,
        window_size_y,
        feature_size=12,
        num_layers=1,
        num_heads=6,
        feedforward_dim=64,
        dropout=0.1,
        output_size=12,
    ):
        super(TimeSeriesTransformer, self).__init__()

        self.gap_prediction = gap_prediction
        self.window_size_y = window_size_y

        self.feature_size = feature_size

        self.positional_encoding = nn.Parameter(torch.randn(1, 107, feature_size))

        self.transformer = nn.Transformer(
            d_model=feature_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
        )

        self.linear = nn.Linear(feature_size, output_size)

    def forward(self, x, verbose=False):
        # Add positional encoding
        x = x + self.positional_encoding

        if verbose:
            print("1:", x.shape)

        # Transformer expects tgt (target tensor) for supervised tasks, but for our forecasting task
        # We'll use shifted input as target (this is a common trick for time series forecasting)
        tgt = torch.roll(x, shifts=-1, dims=1)

        if verbose:
            print("2:", tgt.shape)

        # Apply transformer
        out = self.transformer(x, tgt)

        if verbose:
            print("3:", out.shape)

        # Use relevant steps from the transformer output
        out = out[:, self.gap_prediction : self.gap_prediction + self.window_size_y, :]

        if verbose:
            print("4:", out.shape)

        # Pass through linear layer
        out = self.linear(out)

        if verbose:
            print("5:", out.shape)

        return out


# def test():
#     model = TimeSeriesTransformer()
#     total_params, trainable_params = count_parameters(model)
#     print(f"Total parameters: {total_params}")
#     print(f"Trainable parameters: {trainable_params}")

#     model = TimeSeriesTransformer()
#     model.train()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     input = T.randn(25, 107, 12)
#     out = model(input, verbose=True)
