import math
import os
from pathlib import Path
import time

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler


class Processor:
    def __init__(self, type, scenario, input_features, target_features):
        self.raw_data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', type, scenario)
        self.processed_data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', type, scenario)

        Path(self.processed_data_dir_path).mkdir(parents=True, exist_ok=True)

        self.train_file_names = [f for f in os.listdir(self.raw_data_dir_path) if f[-3:] == "csv" and f[:5] == "train"]
        self.evaluate_file_names = [f for f in os.listdir(self.raw_data_dir_path) if f[-3:] == "csv" and f[:8] == "evaluate"]
        self.test_file_names = [f for f in os.listdir(self.raw_data_dir_path) if f[-3:] == "csv" and f[:4] == "test"]

        # self.raw_train_file_paths = [os.path.join(self.raw_data_dir_path, f) for f in os.listdir(self.raw_data_dir_path) if f[-3:] == "csv" and f[:5] == "train"]
        # self.raw_evaluate_file_paths = [os.path.join(self.raw_data_dir_path, f) for f in os.listdir(self.raw_data_dir_path) if f[-3:] == "csv" and f[:8] == "evaluate"]
        # self.raw_test_file_paths = [os.path.join(self.raw_data_dir_path, f) for f in os.listdir(self.raw_data_dir_path) if f[-3:] == "csv" and f[:4] == "test"]
        #
        # self.processed_train_file_paths = [os.path.join(self.processed_data_dir_path, f) for f in os.listdir(self.processed_data_dir_path) if f[-3:] == "csv" and f[:5] == "train"]
        # self.processed_evaluate_file_paths = [os.path.join(self.processed_data_dir_path, f) for f in os.listdir(self.processed_data_dir_path) if f[-3:] == "csv" and f[:8] == "evaluate"]
        # self.processed_test_file_paths = [os.path.join(self.processed_data_dir_path, f) for f in os.listdir(self.processed_data_dir_path) if f[-3:] == "csv" and f[:4] == "test"]

        # self.processed_train_df = pd.concat((pd.read_csv(f) for f in self.processed_train_file_paths), ignore_index=True)
        # self.processed_evaluate_df = pd.concat((pd.read_csv(f) for f in self.processed_evaluate_file_paths), ignore_index=True)
        # self.processed_test_df = pd.concat((pd.read_csv(f) for f in self.processed_test_file_paths), ignore_index=True)

        self.input_features = input_features
        self.target_features = target_features

    def get_scalers(self, df):
        scalers = []
        for feature in self.all_features:
            col = df[feature].to_numpy().reshape((-1, 1))
            scaler = MinMaxScaler()
            scaler.fit(col)
            scalers.append(scaler)
        return scalers

    # def get_scalers(self, df):
    #     scalers = dict()
    #     for col in df.columns:
    #         col_df = df[col].to_numpy().reshape((-1, 1))
    #         scaler = MinMaxScaler()
    #         scaler.fit(col_df)
    #         scalers[col] = scaler
    #     return scalers

    def scale_df(self, df):
        time_serials = df.to_numpy()
        scaled_time_serials = []
        for (i, scaler) in enumerate(self.scalers):
            scaled_time_serials.append(scaler.transform(np.reshape(time_serials[:,i], [-1, 1])))
        scaled_time_serials = np.hstack(scaled_time_serials)
        return pd.DataFrame(scaled_time_serials, columns=self.all_features)

    def process_raw(self):
        raw_train_df = pd.concat((pd.read_csv(os.path.join(self.raw_data_dir_path, f)) for f in self.train_file_names), ignore_index=True)
        self.all_features = raw_train_df.columns
        self.scalers = self.get_scalers(raw_train_df)

        for file_name in self.train_file_names:
            df = pd.read_csv(os.path.join(self.raw_data_dir_path, file_name))
            scaled_df = self.scale_df(df)
            scaled_df.to_csv(os.path.join(self.processed_data_dir_path, file_name))

        for file_name in self.evaluate_file_names:
            df = pd.read_csv(os.path.join(self.raw_data_dir_path, file_name))
            scaled_df = self.scale_df(df)
            scaled_df.to_csv(os.path.join(self.processed_data_dir_path, file_name))

        for file_name in self.test_file_names:
            df = pd.read_csv(os.path.join(self.raw_data_dir_path, file_name))
            scaled_df = self.scale_df(df)
            scaled_df.to_csv(os.path.join(self.processed_data_dir_path, file_name))

    def insert_anomalies(self):
        for file_name in self.test_file_names:
            df = pd.read_csv(os.path.join(self.raw_data_dir_path, file_name))
            df_copy = df.copy()
            anomalous_df, anomalies_df = self._insert_anomaly(df_copy, sametime_anomalies=False)

            scaled_anomalous_df = self.scale_df(anomalous_df)

            df.to_csv(os.path.join(self.processed_data_dir_path, file_name))
            scaled_anomalous_df.to_csv(os.path.join(self.processed_data_dir_path, f'attack{file_name[4:]}'))
            anomalies_df.to_csv(os.path.join(self.processed_data_dir_path, f'anomalies{file_name[4:]}'))



    def _insert_anomaly(self, df, **kwargs):
        anomalous_df = df.copy()
        global_mins = anomalous_df.min(axis=0)
        global_maxs = anomalous_df.max(axis=0)

        random.seed(kwargs.get("seed", time.time()))
        total_length = anomalous_df.shape[0]

        anomalous_features = kwargs.get("anomalous_features", anomalous_df.columns)
        # anomalous_feature_num = kwargs.get("anomalous_feature_num", len(anomalous_features)//2)
        anomaly_num = kwargs.get("anomaly_num", 10)
        sametime_anomalies = kwargs.get("sametime_anomalies", True)
        max_anomaly_duration = kwargs.get("max_anomaly_duration", 50)
        min_anomaly_duration = kwargs.get("min_anomaly_duration", 10)
        anomaly_interval = kwargs.get("anomaly_interval", int(0.8 * (total_length//anomaly_num - max_anomaly_duration)))

        # chosen_features = []
        # while len(chosen_features) < anomalous_feature_num:
        #     choice = random.choice(anomalous_features)
        #     if choice not in chosen_features:
        #         chosen_features.append(choice)

        anomalies = []

        if sametime_anomalies:
            pass
        else:
            start, end = random.randint(max_anomaly_duration, anomaly_interval), 0
            while len(anomalies) < anomaly_num and start < total_length:
                end = start + random.randint(min_anomaly_duration, max_anomaly_duration)
                duration = end - start
                choice = random.choice(anomalous_features)

                target_df = anomalous_df[choice].iloc[start:end]

                current_var = np.var(target_df)
                previous_var = np.var(anomalous_df[choice].iloc[start-duration:start])

                if current_var < 1 and previous_var < 1:
                    anomaly_type = 'bias'
                elif current_var < 1:
                    anomaly_type = random.choice(['bias', 'replay'])
                elif previous_var < 1:
                    anomaly_type = random.choice(['bias', 'delay'])
                else:
                    anomaly_type = random.choice(['bias', 'delay', 'replay'])

                if anomaly_type == 'bias':
                    downwards = target_df.min() - global_mins[choice]
                    upwards = global_maxs[choice] - target_df.max()

                    if upwards > downwards:
                        bias = random.random() * 0.5 * upwards + 0.25 * upwards
                    else:
                        bias = -1 * random.random() * 0.5 * downwards + 0.25 * downwards

                    target_df += bias
                    anomalies.append([start, end, choice, 'bias', bias])
                elif anomaly_type == 'delay':
                    delay = random.randint(math.floor(0.2 * duration), math.ceil(0.6 * duration))
                    anomalous_df[choice].iloc[start:end] = anomalous_df[choice].iloc[start:end]
                    anomaly_serials[current_step + delay:current_step + duration, current_feature] = serials[
                                                                                                     current_step:current_step + duration - delay,
                                                                                                     current_feature]
                    anomalies.append([current_step, duration, current_feature, 'delay', delay])
                else:
                    pass

                start = end
        return anomalous_df, pd.DataFrame(anomalies, columns=['Start', 'End', 'Feature', 'Type', 'Value'])


def test():
    p = Processor('simulator', 'aircraft_pitch', None, None)
    p.process_raw()
    p.insert_anomalies()
test()