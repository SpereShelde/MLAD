import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from keras_multi_head import MultiHead
import pandas as pd

tf.keras.backend.set_floatx('float64')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from sklearn.preprocessing import MinMaxScaler


def sample_from_csv(file_name, scalers=None, window_size=50, interval=10, jump=0, check=False):
    df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'data', 'testbed', 'barnes', file_name))
    try:
        df.drop(['Reference'], inplace=True, axis=1)
    except KeyError as e:
        pass

    feature_names = df.columns
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
    # print(normalized_time_serials.shape)
    inputs = []
    targets = []

    if interval == 0:
        interval = 1
    for i in range(0, normalized_time_serials.shape[0] - window_size - jump - 1, interval):
        a_sample = normalized_time_serials[i:i + window_size]
        inputs.append(a_sample)
        targets.append([normalized_time_serials[i + jump + window_size, remain_feature_ids]])

    inputs = np.array(inputs)
    targets = np.array(targets)

    # print(inputs.shape)
    # exit(0)

    if check:
        print(f'input shape {inputs.shape}')
        print(f'target shape {targets.shape}')
        print(f'normalized_time_serials shape {normalized_time_serials.shape}')
        # print(inputs[100])
        # for sc in scalers.values():
        #     print(sc.data_range_)

    return inputs, targets, normalized_time_serials, list(zip(remain_feature_ids,
                                                              [feature_names[i] for i in range(len(feature_names))
                                                               if i in remain_feature_ids])), scalers


def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.subtract(tf.reshape(y_true, [-1, y_true.shape[2]]), y_pred)), axis=1)


def lstm(window_len=50, sample_interval=10,
         jump=0, batch_size=64, epochs_num=20,
         attention=False, attn_layer=12,
         dense_dim=32, cell_num=128, bi=True,
         test_only=False, data_file="20181117_Driver1_Trip7.hdf", scenario='bare', plot_loss=False):
    train_inputs, train_targets, train_normalized_time_serials, features, train_scalers = sample_from_csv(
        data_file, window_size=window_len, interval=sample_interval, jump=jump,
        check=True)
    # test_inputs, test_targets, test_normalized_time_serials, _, _ = sample_from_csv('outside_at_end.csv', window_size=monitor_window_length, interval=0, jump=target_skip_steps, check=True, ignore_control=ignore_control, scalers=train_scalers)
    test_inputs = train_inputs[-2000:]
    test_targets = train_targets[-2000:]
    test_normalized_time_serials = train_normalized_time_serials[-2000:]
    train_inputs = train_inputs[:-2000]
    train_targets = train_targets[:-2000]
    train_normalized_time_serials = train_normalized_time_serials[:-2000]

    input_feature_size = train_inputs.shape[2]
    target_feature_size = train_targets.shape[2]

    # inputs_train_diff_min = np.min(np.max(train_inputs, axis=1) - np.min(train_inputs, axis=1), axis=1)
    # sample_weights = np.select([inputs_train_diff_min < 0.05, inputs_train_diff_min < 0.1, inputs_train_diff_min >= 0.1], [1, 2, 3])

    # prin_input(inputs_train, targets_train, sample_weights, feature_size, monitor_window_length, target_skip_steps, rows=8, save=False, show=True)

    model = keras.Sequential()
    a_layer = layers.LSTM(cell_num, dropout=0.05, return_sequences=False)
    if bi:
        a_layer = layers.Bidirectional(a_layer)
    if attention:
        a_layer = MultiHead(a_layer, layer_num=attn_layer)
    model.add(a_layer)
    if attention:
        model.add(layers.Flatten())
    model.add(layers.Dense(dense_dim))
    model.add(layers.Dense(target_feature_size))
    model.build(input_shape=(batch_size, window_len, train_inputs.shape[2]))
    model.compile(optimizer='adam', loss=loss_fn)

    model_name = f'{cell_num}cell-{"Bi" if bi else "Uni"}-LSTM-{f"{attn_layer}lyrs-Attn-" if attention else ""}wl{window_len}-jp{jump}-{dense_dim}/checkpoint'
    model_path = os.path.join(ROOT_DIR, "models", "testbed_models", scenario, model_name)

    if os.path.exists(model_path):
        print("Load Model")
        model.load_weights(model_path)

    if not test_only:
        model.fit(train_inputs, train_targets, validation_split=0.2, batch_size=batch_size, epochs=epochs_num,
                  shuffle=True)
        # model.fit(inputs_train, targets_train, validation_split=0.2, batch_size=batch_size, epochs=epochs_num, shuffle=True, sample_weight=sample_weights)
        model.save_weights(model_path)

    res = model.evaluate(test_inputs, test_targets)
    print(res)

    plot_length = 200

    outputs = []
    for i in range(0, test_inputs.shape[0], 100):
        outputs.append(model(test_inputs[i:i+100, :, :]))
    outputs = np.vstack(outputs)

    title = f'{scenario} {cell_num}cells {"Bi" if bi else "Uni"} LSTM {f"{attn_layer}lyrs Attn" if attention else ""} - {dense_dim} - {window_len}TS - Jump{jump} - Res-{res:.5f}'

    if plot_loss:
        plt.figure(figsize=(30, 16))
        plt.suptitle(f'loss - {title}', fontsize=30)
        targets_test = test_targets.reshape([test_targets.shape[0], test_targets.shape[2]])
        losses = np.abs(outputs - targets_test)
        for i in range(target_feature_size):
            plt.subplot(math.ceil(target_feature_size / 2), 2, i + 1)
            feature_loss = losses[:, i]
            plt.hist(x=feature_loss, bins='auto')
            plt.grid(axis='y')
            plt.title(f'{features[i]}')
        for i in range(1, 10):
            file_name = f'loss - {title}.png'
            if not os.path.exists(os.path.join(ROOT_DIR, "results", 'testbed', scenario, file_name)):
                plt.savefig(os.path.join(ROOT_DIR, "results", 'testbed', scenario, file_name))
                # plt.show()
                break

    plt.figure(figsize=(30, 16))
    plt.suptitle(title, fontsize=30)
    plot_length = min(400, ((test_inputs.shape[0] - window_len - jump) // 100) * 100)
    x = np.arange(plot_length).reshape(-1, 1)

    for i in range(len(features)):
        plt.subplot(math.ceil(target_feature_size / 2), 2, i + 1)
        plt.plot(x, np.array(outputs[:plot_length, i]).reshape(-1, 1), label='predict', c='r', marker='.')
        plt.plot(x, np.array(test_targets[:plot_length, 0, i]).reshape(-1, 1), label='target', c='b', marker='.')
        plt.title(f'{features[i][1]}')

        # sorted_outputs = np.array(list(map(list, zip(*flat_results[:plot_length])))[0])
        # sorted_targets = np.array(list(map(list, zip(*flat_results[:plot_length])))[1])
        # for i in range(feature_size):
        #     plt.subplot(4, math.ceil(feature_size / 2), feature_size + i + 1)
        #     plt.scatter(x, sorted_outputs[:, i].reshape(1, -1), label='predict', c='r', marker='+')
        #     plt.scatter(x, sorted_targets[:, 0, i].reshape(1, -1), label='target', c='b', marker='x')
        #     plt.title(f'{feature_names[i]}')
    for i in range(1, 10):
        file_name = f'{title}.png'
        if not os.path.exists(os.path.join(ROOT_DIR, "results", 'testbed', scenario, file_name)):
            plt.savefig(os.path.join(ROOT_DIR, "results", 'testbed', scenario, file_name))
            # plt.show()
            break

scenario = 'pid_kf'
data_file = 'canvas_semi_auto_pid_kf.csv'

lstm(window_len=50, sample_interval=1,
     jump=0, batch_size=64, epochs_num=50,
     bi=False, attention=False, attn_layer=0,
     cell_num=128, dense_dim=64,
     test_only=True, data_file=data_file, scenario=scenario, plot_loss=True)
lstm(window_len=50, sample_interval=1,
     jump=0, batch_size=64, epochs_num=50,
     bi=True, attention=False, attn_layer=0,
     cell_num=128, dense_dim=64,
     test_only=False, data_file=data_file, scenario=scenario, plot_loss=True)
lstm(window_len=50, sample_interval=1,
     jump=0, batch_size=64, epochs_num=50,
     bi=False, attention=True, attn_layer=1,
     cell_num=128, dense_dim=64,
     test_only=False, data_file=data_file, scenario=scenario, plot_loss=True)
lstm(window_len=50, sample_interval=1,
     jump=0, batch_size=64, epochs_num=50,
     bi=True, attention=True, attn_layer=1,
     cell_num=128, dense_dim=64,
     test_only=False, data_file=data_file, scenario=scenario, plot_loss=True)
lstm(window_len=50, sample_interval=1,
     jump=0, batch_size=64, epochs_num=50,
     bi=True, attention=True, attn_layer=2,
     cell_num=128, dense_dim=64,
     test_only=False, data_file=data_file, scenario=scenario, plot_loss=True)
lstm(window_len=50, sample_interval=1,
     jump=0, batch_size=64, epochs_num=80,
     bi=True, attention=True, attn_layer=4,
     cell_num=128, dense_dim=64,
     test_only=False, data_file=data_file, scenario=scenario, plot_loss=True)
