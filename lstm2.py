class VectorOutputLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, n_future, device):
        super(VectorOutputLSTM, self).__init__()

        self.device = device

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim * n_future)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=self.device).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=self.device).requires_grad_()

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.linear(out[:, -1, :])

        # Reshape to get predictions for the future time steps
        return out.view(x.size(0), -1)


model = VectorOutputLSTM(12, 64, 1, 5, WINDOW_SIZE_Y + GAP_PREDICTION, "cuda").to("cuda")
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.001)
input = T.randn(25, 107, 12, device="cuda")
out = model(input)
print(out.shape)
