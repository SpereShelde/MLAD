import math
import random
from os import listdir
from os.path import isfile

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from classic_rnn.TemporalCrudeAttention import TemporalCrudeAttention
import os
from DatasetReader import DatasetReader
from keras_multi_head import MultiHead
import pandas as pd
tf.keras.backend.set_floatx('float64')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from sklearn.preprocessing import MinMaxScaler

def sample_from_csv(file_name, scalers = None, window_size = 50, interval = 10, jump = 0, check=False, ignore_control = True):
    df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'data', 'testbed', file_name))
    feature_names = df.columns
    if ignore_control:
        omit_feature_ids = []
        for name in ['Throttle_Control', 'Servo_Control', 'Voltage', 'Throttle_A', 'Throttle_B', 'Servo', 'Linear_Y']:
            omit_feature_ids.append(df.columns.get_loc(name))
        remain_feature_ids = [id for id in range(len(feature_names)) if id not in omit_feature_ids]
        # print(omit_features_id)
        # print(remain_feature_ids)
        # exit(0)

    normalized_time_serials = []
    if not scalers or len(scalers) == 0:
        scalers = dict()
        for col in df.columns:
            col_df = df[col].to_numpy().reshape((-1, 1))
            scaler = MinMaxScaler()
            scaler.fit(col_df)
            scalers[col] = scaler
            normalized_time_serials.append(scaler.transform(col_df))
    else:
        for col in df.columns:
            col_df = df[col].to_numpy().reshape((-1, 1))
            scaler = scalers[col]
            normalized_time_serials.append(scaler.transform(col_df))
    normalized_time_serials = np.hstack(normalized_time_serials)

    inputs = []
    targets = []

    if interval == 0:
        interval = 1
    for i in range(0, normalized_time_serials.shape[0] - window_size - jump - 1, interval):
        a_sample = normalized_time_serials[i:i + window_size]
        inputs.append(a_sample)
        if ignore_control:
            targets.append([normalized_time_serials[i + jump + window_size, remain_feature_ids]])
        else:
            targets.append([normalized_time_serials[i + jump + window_size]])

    inputs = np.array(inputs)
    targets = np.array(targets)

    if check:
        print(f'input shape {inputs.shape}')
        print(f'target shape {targets.shape}')
        print(f'normalized_time_serials shape {normalized_time_serials.shape}')
        # print(inputs[100])
        # for sc in scalers.values():
        #     print(sc.data_range_)

    if ignore_control:
        # return inputs, targets, normalized_time_serials, remain_feature_ids, scalers
        return inputs, targets, normalized_time_serials, list(zip(remain_feature_ids, [feature_names[i] for i in range(len(feature_names)) if i in remain_feature_ids])), scalers
    else:
        return inputs, targets, normalized_time_serials, feature_names, scalers

def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.subtract(tf.reshape(y_true, [-1, y_true.shape[2]]), y_pred)), axis=1)

def multi_time_serial_lstm_transformer_attention(monitor_window_length = 50, window_sample_interval = 10, target_skip_steps = 0, batch_size = 64, epochs_num = 20, layer_num = 12, ignore_control = True, save_img = True):
    train_inputs, train_targets, train_normalized_time_serials, features, _ = sample_from_csv('complete.csv', ignore_control=ignore_control)
    test_inputs, test_targets, test_normalized_time_serials, _, _ = sample_from_csv('outside_at_end.csv', interval=0, ignore_control=ignore_control)


    input_feature_size = train_inputs.shape[2]
    target_feature_size = train_targets.shape[2]

    plot_length = min(500, ((train_inputs.shape[0] - monitor_window_length - target_skip_steps) // 100) * 100)

    # inputs_train_diff_min = np.min(np.max(train_inputs, axis=1) - np.min(train_inputs, axis=1), axis=1)
    # sample_weights = np.select([inputs_train_diff_min < 0.05, inputs_train_diff_min < 0.1, inputs_train_diff_min >= 0.1], [1, 2, 3])

    # prin_input(inputs_train, targets_train, sample_weights, feature_size, monitor_window_length, target_skip_steps, rows=8, save=False, show=True)


    dense_1_num = 256
    dense_2_num = 64

    if ignore_control:
        model_name = f'{monitor_window_length}-{target_skip_steps}-{layer_num}layers-{dense_1_num}-{dense_2_num}-wout_ctrl/checkpoint'
    else:
        model_name = f'{monitor_window_length}-{target_skip_steps}-{layer_num}layers-{dense_1_num}-{dense_2_num}-with_ctrl/checkpoint'

    model_path = os.path.join(ROOT_DIR, "models", "testbed_models", model_name)

    model = keras.Sequential()
    model.add(MultiHead(layers.Bidirectional(layers.LSTM(128, return_sequences=False)), layer_num=layer_num))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_1_num))
    model.add(layers.Dense(dense_2_num))
    model.add(layers.Dense(target_feature_size))

    model.build(input_shape=(batch_size, monitor_window_length, train_inputs.shape[2]))
    model.compile(optimizer='adam', loss=loss_fn)
    # model.summary()
    # exit(0)

    if os.path.exists(model_path):
        print("Load Model")
        model.load_weights(model_path)

    model.fit(train_inputs, train_targets, validation_split=0.2, batch_size=batch_size, epochs=epochs_num, shuffle=True)
    # model.fit(train_inputs, train_targets, validation_split=0.2, batch_size=batch_size, epochs=epochs_num, shuffle=True, sample_weight=sample_weights)

    model.save_weights(model_path)

    res = model.evaluate(test_inputs, test_targets)
    print(res)

    ploted = 0
    outputs = []
    while ploted < plot_length:
        to_plot = min(plot_length, ploted+100)
        outputs.append(model(test_inputs[ploted:to_plot,:,:]))
        ploted += 100
    outputs = np.vstack(outputs)
    plot_length = outputs.shape[0]
    x = np.arange(plot_length).reshape(-1, 1)

    plt.figure(figsize=(30, 16))
    plt.suptitle(f'{target_feature_size} Features LSTM {layer_num} Layers Trans Attn - {monitor_window_length} TimeSteps - Jump {target_skip_steps} Steps - {window_sample_interval}StepInterval-{batch_size}Batch-{epochs_num}Epochs-Res-{res}', fontsize=30)

    for i in range(len(features)):
        plt.subplot(math.ceil(target_feature_size/2), 2, i + 1)
        plt.plot(x, np.array(outputs[:, i]).reshape(-1, 1), label='predict', c='r', marker='.')
        plt.plot(x, np.array(test_normalized_time_serials[monitor_window_length+target_skip_steps:monitor_window_length+target_skip_steps+plot_length, features[i][0]]).reshape(-1, 1), label='target', c='b', marker='.')
        plt.title(f'{features[i][1]}')
        # plt.xlim([900, 1000])

    if not save_img:
        plt.show()
    else:

        # sorted_outputs = np.array(list(map(list, zip(*flat_results[:plot_length])))[0])
        # sorted_targets = np.array(list(map(list, zip(*flat_results[:plot_length])))[1])
        # for i in range(feature_size):
        #     plt.subplot(4, math.ceil(feature_size / 2), feature_size + i + 1)
        #     plt.scatter(x, sorted_outputs[:, i].reshape(1, -1), label='predict', c='r', marker='+')
        #     plt.scatter(x, sorted_targets[:, 0, i].reshape(1, -1), label='target', c='b', marker='x')
        #     plt.title(f'{feature_names[i]}')
        for i in range(1, 10):
            file_name = f"trans-attn-fs{target_feature_size}-ws{monitor_window_length}-jump{target_skip_steps}-res{res}-{i}.png"
            if not os.path.exists(os.path.join(ROOT_DIR, "results", file_name)):
                plt.savefig(os.path.join(ROOT_DIR, "results", file_name))
                # plt.show()
                break

# todo:
multi_time_serial_lstm_transformer_attention(monitor_window_length=50, window_sample_interval=10, target_skip_steps=0, batch_size=16, epochs_num=1, layer_num=2, ignore_control = True, save_img = False)
