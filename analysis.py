# %% [markdown]
# # Research Question: Day-Ahead consumption forecasting
# Forecast is made at 11:00 for the next day (all 24 hours).
# 1. Domain understanding
# 2. Data acquisition
# 3. Data exploration and visualization
# 4. Data preprocessing: cleaning, transformation, normalization
# 5. Feature engineering: construction, selection
# 6. Model engineering: construction, selection, hyperparameter tuning
# 7. Evaluation: cross-validation, comparing with baselines

# Time series components: Seasonality, Trend, Cyclic, Irregular

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dtype_mapping = {
    "location": "category",
    "consumption": "float",
    "temperature": "float",
}
df = pd.read_csv("data/consumption_temp.csv", dtype=dtype_mapping, parse_dates=["time"])
# %%
df.info()
# %%
df.head()
# %%
# get counts of each location
df["location"].value_counts()
# %%
# drop location helsingfors
df = df[df["location"] != "helsingfors"]
df["location"] = df["location"].cat.remove_unused_categories()
df["location"].value_counts()
# %%
# explode data based on location
# Pivot for consumption
df_consumption = df.pivot(index="time", columns="location", values="consumption")
df_consumption.columns = [f"{col}_consumption" for col in df_consumption.columns]

# Pivot for temperature
df_temperature = df.pivot(index="time", columns="location", values="temperature")
df_temperature.columns = [f"{col}_temp" for col in df_temperature.columns]

# Merge the two dataframes on time
df_merged = df_consumption.merge(df_temperature, left_index=True, right_index=True)
df_merged.head()
# %%
# add hour column
df_merged["hour"] = df_merged.index.hour
# add weekday column
df_merged["weekday"] = df_merged.index.weekday
df_merged.head()
# %%
# standardize data
np_df = df_merged.to_numpy()
# get mean and std for consumption
consumption_mean = np_df[:, 0:5].mean()
consumption_std = np_df[:, 0:5].std()

# get mean and std for temperature
temp_mean = np_df[:, 5:10].mean()
temp_std = np_df[:, 5:10].std()

# standardize data
np_df[:, 0:5] = (np_df[:, 0:5] - consumption_mean) / consumption_std
np_df[:, 5:10] = (np_df[:, 5:10] - temp_mean) / temp_std
np_df


# %%
# create a function to makes splits of the data
def split_data(matrix, window_size_x, gap_prediction, window_size_y, start_index):
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


# %%
WINDOW_SIZE_X = 107
GAP_PREDICTION = 13
WINDOW_SIZE_Y = 24
# %%
# use function to create dataset
X, y = split_data(
    np_df,
    window_size_x=WINDOW_SIZE_X,
    gap_prediction=GAP_PREDICTION,
    start_index=1,
    window_size_y=WINDOW_SIZE_Y,
)
X.shape
# %%
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
X_train.shape, y_train.shape, X_test.shape, y_test.shape
# %%
