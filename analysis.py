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
