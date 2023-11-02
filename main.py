import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from lstm1 import TimeSeriesLSTM
from lstm2 import VectorOutputLSTM
from transformer1 import TimeSeriesTransformer
from transformer2 import TransformerForecasting

WINDOW_SIZE_X = 107
GAP_PREDICTION = 13
WINDOW_SIZE_Y = 24


def load_data() -> pd.DataFrame:
    # mapping to save space and time with dataset
    dtype_mapping = {
        "location": "category",
        "consumption": "float",
        "temperature": "float",
    }
    df = pd.read_csv("data/consumption_temp.csv", dtype=dtype_mapping, parse_dates=["time"])

    # drop location helsingfors
    df = df[df["location"] != "helsingfors"]
    df["location"] = df["location"].cat.remove_unused_categories()

    # Pivot for consumption
    df_consumption = df.pivot(index="time", columns="location", values="consumption")
    df_consumption.columns = [f"{col}_consumption" for col in df_consumption.columns]
    # Pivot for temperature
    df_temperature = df.pivot(index="time", columns="location", values="temperature")
    df_temperature.columns = [f"{col}_temp" for col in df_temperature.columns]
    # Merge the two dataframes on time
    df_merged = df_consumption.merge(df_temperature, left_index=True, right_index=True)

    # add hour column
    df_merged["hour"] = df_merged.index.hour
    # add weekday column
    df_merged["weekday"] = df_merged.index.weekday

    return df_merged


def standardize_data() -> np.ndarray:
    # standardize data
    np_df = df_merged.to_numpy()
    # get mean and std for consumption
    consumption_mean = np_df[:, 0:5].mean()
    consumption_std = np_df[:, 0:5].std()
    # get mean and std for temperature
    temp_mean = np_df[:, 5:10].mean()
    temp_std = np_df[:, 5:10].std()

    np_df[:, 0:5] = (np_df[:, 0:5] - consumption_mean) / consumption_std
    np_df[:, 5:10] = (np_df[:, 5:10] - temp_mean) / temp_std

    return np_df


# create a function to makes splits of the data
def split_data(matrix, window_size_x, gap_prediction, window_size_y, start_index) -> np.ndarray:
    # df: dataframe to split
    # window_size_x: size of the window
    # gap_prediction: how many time steps between last X value and the y value
    # gap_start: how many time steps until first X value
    Xs = []
    ys = []
    # loop through the data
    for i in range(start_index, matrix.shape[0] - window_size_x - gap_prediction - window_size_y):
        # get the start and end of the prediction gap
        pred = i + window_size_x + gap_prediction
        # get the X and y
        X = matrix[i : i + window_size_x]
        y = matrix[pred : pred + window_size_y]
        # add to the lists
        Xs.append(X)
        ys.append(y)
    # return the split data
    return np.array(Xs), np.array(ys)


def split_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # use function to create dataset
    X, y = split_data(
        np_df,
        window_size_x=WINDOW_SIZE_X,
        gap_prediction=GAP_PREDICTION,
        start_index=1,
        window_size_y=WINDOW_SIZE_Y,
    )

    # convert to train and test sets
    train_split = int(0.8 * len(X))
    # split the data randomly
    np.random.seed(1)
    train_indices = np.random.choice(range(len(X)), size=train_split, replace=False)
    test_indices = list(set(range(len(X))) - set(train_indices))
    # create train and test sets
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    # check shapes
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test


def count_parameters(model) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# --------------------------------------------


# Training loop
def train_model_type1(model, train_X, train_y, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_X)

        outputs = outputs[:, :, :5]
        train_y = train_y[:, :, :5]

        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


# Evaluation loop
def evaluate_model_type1(model, test_X, test_y, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(test_X)

        outputs = outputs[:, :, :5]
        test_y = test_y[:, :, :5]

        loss = criterion(outputs, test_y)
    return loss.item()


def train_and_eval1(model_type: str):
    # Instantiate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "lstm":
        model = TimeSeriesLSTM().to(device)
    elif model_type == "transformer":
        model = TimeSeriesTransformer().to(device)
    else:
        raise ValueError("Invalid model_type")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert them to tensors
    train_X = torch.tensor(X_train, dtype=torch.float32).to(device)
    train_y = torch.tensor(y_train, dtype=torch.float32).to(device)
    test_X = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_y = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Train the model
    train_model_type1(model, train_X, train_y, criterion, optimizer, epochs=1000)

    # Evaluate the model
    test_loss = evaluate_model1(model, test_X, test_y, criterion)
    print(f"Test Loss: {test_loss:.4f}")

    # Predict using the model
    model.eval()
    with torch.no_grad():
        predictions = model(test_X)

    # Choose a random sample from the test set
    sample_idx = torch.randint(0, test_X.size(0), (1,)).item()
    sample_true = test_y[sample_idx].numpy(force=True)
    sample_pred = predictions[sample_idx].numpy(force=True)

    # Plotting
    plt.figure(figsize=(14, 6))

    for i in range(12):  # For each of the 12 features
        plt.subplot(3, 4, i + 1)
        plt.plot(sample_true[:, i], label="True", marker="o")
        plt.plot(sample_pred[:, i], label="Predicted", marker="x")
        plt.title(f"{df_merged.columns[i]}")
        plt.legend()

    plt.tight_layout()
    plt.show()


# --------------------------------------------


# Training loop
def train_model2(model, train_X, train_y, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_X).view(-1, WINDOW_SIZE_Y + GAP_PREDICTION, 5)[:, :WINDOW_SIZE_Y, :]
        # print(outputs.shape)

        train_y = train_y[:, :, :5]
        # print(train_y.shape)

        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


# Evaluation loop
def evaluate_model2(model, test_X, test_y, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(test_X)

        outputs = outputs.view(-1, WINDOW_SIZE_Y + GAP_PREDICTION, 5)[:, :WINDOW_SIZE_Y, :]

        test_y = test_y[:, :, :5]

        loss = criterion(outputs, test_y)
    return loss.item()


def train_and_eval2(model_type: str):
    # Instantiate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "lstm":
        model = VectorOutputLSTM(12, 64, 1, 5, WINDOW_SIZE_Y + GAP_PREDICTION, "cuda").to(device)
    elif model_type == "transformer":
        model = TransformerForecasting(12, 64, 2, 3, 5, WINDOW_SIZE_Y + GAP_PREDICTION).to(device)
    else:
        raise ValueError("Invalid model_type")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model2(model, train_X, train_y, criterion, optimizer, epochs=1000)

    # Evaluate the model
    test_loss = evaluate_model2(model, test_X, test_y, criterion)
    print(f"Test Loss: {test_loss:.4f}")

    # Predict using the model
    model.eval()
    with torch.no_grad():
        predictions = model(test_X).view(-1, WINDOW_SIZE_Y + GAP_PREDICTION, 5)[:, :WINDOW_SIZE_Y, :]

    # Choose a random sample from the test set
    sample_idx = torch.randint(0, test_X.size(0), (1,)).item()
    sample_true = test_y[sample_idx].numpy(force=True)
    sample_pred = predictions[sample_idx].numpy(force=True)

    # Plotting
    plt.figure(figsize=(14, 6))

    for i in range(5):  # For each of the 12 features
        plt.subplot(3, 4, i + 1)
        plt.plot(sample_true[:, i], label="True", marker="o")
        plt.plot(sample_pred[:, i], label="Predicted", marker="x")
        plt.title(f"{df_merged.columns[i]}")
        plt.legend()

    plt.tight_layout()
    plt.show()


# --------------------------------------------


def tsai_run():
    from tsai.all import *

    bs = 64
    c_in = 12  # aka channels, features, variables, dimensions
    c_out = 5
    seq_len = 107

    xb = torch.randn(bs, c_in, seq_len)

    # standardize by channel by_var based on the training set
    xb = (xb - xb.mean((0, 2), keepdim=True)) / xb.std((0, 2), keepdim=True)

    # Settings
    max_seq_len = 256
    d_model = 128
    n_heads = 6
    d_k = d_v = None  # if None --> d_model // n_heads
    d_ff = 256
    dropout = 0.1
    n_layers = 3
    fc_dropout = 0.1
    kwargs = {}

    model = TST(
        c_in,
        c_out,
        seq_len,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
        dropout=dropout,
        n_layers=n_layers,
        fc_dropout=fc_dropout,
        **kwargs,
    )
    test_eq(model.to(xb.device)(xb).shape, [bs, c_out])
    print(f"model parameters: {count_parameters(model)}")

    print(model(T.rand((10, 12, 107))).shape)


def main():
    train_and_eval1("lstm")
    train_and_eval2("lstm")
    train_and_eval1("transformer")
    train_and_eval2("transformer")
    # tsai_run()


if __name__ == "__main__":
    main()
