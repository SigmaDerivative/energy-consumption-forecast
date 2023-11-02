class TransformerForecasting(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, output_dim, n_future):
        super(TransformerForecasting, self).__init__()

        # Embedding layer: This layer converts the input features into embeddings of dimension d_model
        self.embedding = nn.Linear(input_dim, d_model)

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_encoder_layers,
        )

        # Output layer: Produces a vector for future time steps
        self.output_layer = nn.Linear(d_model, output_dim * n_future)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.output_layer(x[:, -1, :])  # Only consider the last time step's output for forecasting
        return x.view(x.size(0), -1)


model = TransformerForecasting(12, 64, 2, 3, 5, WINDOW_SIZE_Y + GAP_PREDICTION).to("cuda")
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.001)
input = T.randn(25, 107, 12, device="cuda")
out = model(input)
print(out.shape)
