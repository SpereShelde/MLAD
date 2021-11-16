import math
import random
from os import listdir
from os.path import isfile

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

from tools.DatasetReader import DatasetReader
from tools.anomaly_creator import insert_super_anomalies

tf.keras.backend.set_floatx('float64')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_files = [
    "20181113_Driver1_Trip1.hdf", "20181113_Driver1_Trip2.hdf", "20181116_Driver1_Trip3.hdf",
    "20181116_Driver1_Trip4.hdf", "20181116_Driver1_Trip5.hdf", "20181116_Driver1_Trip6.hdf",
    "20181117_Driver1_Trip8.hdf", "20181203_Driver1_Trip9.hdf", "20181203_Driver1_Trip10.hdf"
]

selected_features = ['/CAN/Yawrate1', '/CAN/ENG_Trq_DMD', '/CAN/VehicleSpeed', '/CAN/AccPedal', '/CAN/ENG_Trq_ZWR',
                     '/Plugins/Velocity_X', '/GPS/Direction', '/CAN/EngineTemperature']
speed_features = ['/CAN/AccPedal', '/CAN/EngineSpeed_CAN', '/CAN/VehicleSpeed', '/GPS/Velocity',
                  '/Plugins/Accelerometer_X', '/Plugins/Body_acceleration_X', '/Plugins/Velocity_X',
                  '/CAN/WheelSpeed_RL']
corelated_features = ['/CAN/AccPedal', '/CAN/ENG_Trq_DMD', '/CAN/ENG_Trq_ZWR', '/CAN/ENG_Trq_m_ex',
                      '/CAN/EngineSpeed_CAN', '/CAN/OilTemperature1', '/CAN/Trq_Indicated', '/CAN/VehicleSpeed',
                      '/CAN/WheelSpeed_FL', '/CAN/WheelSpeed_FR', '/CAN/WheelSpeed_RL', '/CAN/WheelSpeed_RR']

sub_corelated_features = ['/CAN/AccPedal', '/CAN/ENG_Trq_ZWR', '/CAN/ENG_Trq_m_ex',
                      '/CAN/EngineSpeed_CAN', '/CAN/Trq_Indicated', '/CAN/VehicleSpeed',
                      '/CAN/WheelSpeed_FL', '/CAN/WheelSpeed_FR', '/CAN/WheelSpeed_RL', '/CAN/WheelSpeed_RR']

def ae(test_file, batch_size=64, epochs_num=10,
         feature_set='corelated_features', features=None):
    data_reader = DatasetReader(train_files + [test_file])

    scalers = data_reader.get_scalers(train_files, feature_names=features)

    train_serials, _ = data_reader._concatenate_data(file_names=train_files, feature_names=features)
    test_serials, _ = data_reader._concatenate_data(file_names=[test_file], feature_names=features)

    anomaly_serials, anomalies = insert_super_anomalies(test_serials, 100, 50)
    while len(anomalies) == 0:
        anomaly_serials, anomalies = insert_super_anomalies(test_serials, 100, 50)

    normalized_train_serials = []
    normalized_test_serials = []
    normalized_anomaly_serials = []

    for i in range(len(scalers)):
        scaler = scalers[i]
        normalized_train_serials.append(scaler.transform(train_serials[:, i]))
        normalized_test_serials.append(scaler.transform(test_serials[:, i]))
        normalized_anomaly_serials.append(scaler.transform(anomaly_serials[:, i]))

    normalized_train_serials = np.hstack(normalized_train_serials)
    normalized_test_serials = np.hstack(normalized_test_serials)
    normalized_anomaly_serials = np.hstack(normalized_anomaly_serials)

    print(f'normalized_train_serials: {normalized_train_serials.shape}')
    print(f'normalized_test_serials: {normalized_test_serials.shape}')
    print(f'normalized_anomaly_serials shape: {normalized_anomaly_serials.shape}')
    print('anomalies:')
    print(anomalies)

    input_layer = layers.Input(shape=(16, ))

    encoder = layers.Dense(12, activation="tanh")(input_layer)
    encoder = layers.Dense(8, activation="relu")(encoder)
    decoder = layers.Dense(12, activation='tanh')(encoder)
    decoder = layers.Dense(16, activation='relu')(decoder)

    autoencoder = keras.Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mean_squared_logarithmic_error',metrics=['accuracy'])

    model_name = f'AE'
    model_weight_path = os.path.join(ROOT_DIR, 'models_weights', feature_set, f'{model_name}/checkpoint')

    if os.path.exists(model_weight_path):
        print("Load Model")
        autoencoder.load_weights(model_weight_path)

    if epochs_num > 0:
        autoencoder.fit(normalized_train_serials, normalized_train_serials,
                        epochs=epochs_num,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1)
        autoencoder.save_weights(model_weight_path)

    res = autoencoder.evaluate(normalized_test_serials, normalized_test_serials)
    print(res)

    outputs = []
    losses = []
    for i in range(len(normalized_anomaly_serials.shape[0])):
        sample = normalized_anomaly_serials[i]
        output = autoencoder(sample)
        loss = np.abs(output - sample)
        outputs.append(output)
        losses.append(loss)


ae("20181117_Driver1_Trip7.hdf", batch_size=64, epochs_num=60,
     feature_set='sub_corelated_features', features=sub_corelated_features)