import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_scalers(df):
    scalers = []
    for feature in df.columns:
        col = df[feature].to_numpy().reshape((-1, 1))
        scaler = MinMaxScaler()
        scaler.fit(col)
        scalers.append(scaler)
    return scalers

def scale_df(df, scalers):
    time_serials = df.to_numpy()
    scaled_time_serials = []
    for (i, scaler) in enumerate(scalers):
        scaled_time_serials.append(scaler.transform(np.reshape(time_serials[:,i], [-1, 1])))
    scaled_time_serials = np.hstack(scaled_time_serials)
    return pd.DataFrame(scaled_time_serials, columns=df.columns)

benign_file_names = [f for f in os.listdir('.') if f[-3:] == "csv" and f[:6] == "benign"]
anomalous_file_names = [f for f in os.listdir('.') if f[-3:] == "csv" and f[:9] == "anomalous"]

raw_train_df = pd.concat((pd.read_csv(f) for f in benign_file_names), ignore_index=True)
scalers = get_scalers(raw_train_df)

for file in benign_file_names:
    df = pd.read_csv(file)
    scaled_df = scale_df(df, scalers)
    scaled_df.to_csv(f'scaled-{file}', index=False)

for file in anomalous_file_names:
    df = pd.read_csv(file)
    scaled_df = scale_df(df, scalers)
    scaled_df.to_csv(f'scaled-{file}', index=False)

