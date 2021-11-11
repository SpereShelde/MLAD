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
from classic_rnn.TemporalCrudeAttention import TemporalCrudeAttention
import os
from tools.DatasetReader import DatasetReader
from keras_multi_head import MultiHead

# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
# os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"
from tools.anomaly_creator import insert_super_anomalies, print_all

tf.keras.backend.set_floatx('float64')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_files = [
    "20181113_Driver1_Trip1.hdf", "20181113_Driver1_Trip2.hdf", "20181116_Driver1_Trip3.hdf",
    "20181116_Driver1_Trip4.hdf", "20181116_Driver1_Trip5.hdf", "20181116_Driver1_Trip6.hdf",
    "20181117_Driver1_Trip8.hdf", "20181203_Driver1_Trip9.hdf", "20181203_Driver1_Trip10.hdf"
]


def compare(item1, item2):
    r1 = abs(item1[0] - item1[1])
    r2 = abs(item2[0] - item2[1])
    if r1 > r2:
        return -1
    elif r1 < r2:
        return 1
    else:
        return 0


def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.subtract(tf.reshape(y_true, [-1, y_true.shape[2]]), y_pred)), axis=1)


def prin_input(inputs_train, targets_train, sample_weights, feature_size, monitor_window_length, target_skip_steps,
               rows=8, save=False, show=True):
    plt.figure(figsize=(30, 16))
    for i in range(rows * rows):
        idx = random.randint(0, inputs_train.shape[0])
        plt.subplot(rows, rows, i + 1)
        color = 'purple'
        if sample_weights[idx] == 1:
            color = 'y'
        elif sample_weights[idx] == 3:
            color = 'b'
        elif sample_weights[idx] == 6:
            color = 'g'
        for j in range(feature_size):
            plt.plot(np.arange(monitor_window_length), inputs_train[idx, :, j], color=color)
            plt.scatter(monitor_window_length + target_skip_steps - 1, targets_train[idx, 0, j], color='r')
        plt.ylim([0, 1])
        plt.xlim([0, monitor_window_length + target_skip_steps])
    if save:
        for i in range(1, 20):
            file_name = f"input-map-fs{feature_size}-{i}.png"
            if not os.path.exists(os.path.join(ROOT_DIR, "results", file_name)):
                plt.savefig(os.path.join(ROOT_DIR, "results", file_name))
                break
    if show:
        plt.show()


# def plot_lost(window_length=50, sample_interval=10, jump=0, batch_size=64, epochs_num=10,
#          feature_set='corelated_features', features=None, cell_num=128, dense_dim=64,
#          bidirection=False, attention=False, attn_layer=4,
#          test_only=False, test_file="20181117_Driver1_Trip7.hdf"):
#     pass

def lstm(window_length=50, sample_interval=10, jump=0, batch_size=64, epochs_num=10,
         feature_set='corelated_features', features=None, cell_num=128, dense_dim=64,
         bidirection=False, attention=False, attn_layer=4,
         test_only=False, test_file="20181117_Driver1_Trip7.hdf", plot_loss=False):
    if not test_only:
        test_file = "20181117_Driver1_Trip7.hdf"

    data_reader = DatasetReader(train_files + [test_file])
    inputs_train, targets_train, inputs_test, targets_test, features, _ = data_reader.sample(features,
                                                                                             time_steps=window_length,
                                                                                             sample_interval=sample_interval,
                                                                                             target_sequence=False,
                                                                                             split_test_from_train=False,
                                                                                             target_skip_steps=jump,
                                                                                             test_files=[test_file])
    print("train input shape", inputs_train.shape)
    print("train target shape", targets_train.shape)
    print("test input shape", inputs_test.shape)
    print("test target shape", targets_test.shape)

    feature_size = inputs_test.shape[2]

    # inputs_train_diff_min = np.min(np.max(inputs_train, axis=1) - np.min(inputs_train, axis=1), axis=1)
    # sample_weights = np.select([inputs_train_diff_min < 0.05, inputs_train_diff_min < 0.1, inputs_train_diff_min >= 0.1], [1, 3, 6])

    # prin_input(inputs_train, targets_train, sample_weights, feature_size, monitor_window_length, target_skip_steps, rows=8, save=False, show=True)

    model = keras.Sequential()
    # model.add(layers.LSTM(cell_num, dropout=0.05, return_sequences=False))
    a_layer = layers.LSTM(cell_num, dropout=0.05, return_sequences=False)
    if bidirection:
        a_layer = layers.Bidirectional(a_layer)
    if attention:
        a_layer = MultiHead(a_layer, layer_num=attn_layer)
    model.add(a_layer)
    if attention:
        model.add(layers.Flatten())
    model.add(layers.Dense(dense_dim))
    model.add(layers.Dense(feature_size))
    model.build(input_shape=(batch_size, window_length, feature_size))
    model.compile(optimizer='adam', loss=loss_fn)

    model_name = f'{cell_num}cell-{"Bi" if bidirection else "Uni"}-LSTM-{f"{attn_layer}lyrs-Attn-" if attention else ""}wl{window_length}-jp{jump}-{dense_dim}'
    model_weight_path = os.path.join(ROOT_DIR, 'models_weights', feature_set, f'{model_name}/checkpoint')
    complete_model_path = os.path.join(ROOT_DIR, 'complete_models', feature_set, model_name)
    if os.path.exists(model_weight_path):
        print("Load Model")
        model.load_weights(model_weight_path)

    if not test_only:
        if epochs_num > 0:
            model.fit(inputs_train, targets_train, validation_split=0.2, batch_size=batch_size, epochs=epochs_num,
                      shuffle=True)
        # model.fit(inputs_train, targets_train, validation_split=0.2, batch_size=batch_size, epochs=epochs_num, shuffle=True, sample_weight=sample_weights)
        model.save_weights(model_weight_path)
        # model.save(complete_model_path)
        # # print(complete_model_path)
        # new_model = keras.models.load_model(complete_model_path, custom_objects={'loss_fn': loss_fn})
        # new_res = new_model.evaluate(inputs_test, targets_test)
        # print(new_res)

    res = model.evaluate(inputs_test, targets_test)
    print(res)

    outputs = []
    for i in range(0, inputs_test.shape[0], 100):
        outputs.append(model(inputs_test[i:i + 100, :, :]))
    outputs = np.vstack(outputs)

    title = f'{feature_size} {feature_set} {cell_num}cells {"Bi" if bidirection else "Uni"} LSTM {f"{attn_layer}lyrs Attn" if attention else ""} - {dense_dim} - {window_length}TS - Jump{jump} - Res-{res:.5f} - {test_file[15:-4]}'

    if plot_loss:
        losses = np.abs(outputs - targets_test.reshape([targets_test.shape[0], targets_test.shape[2]]))
        df = pd.DataFrame(data=losses, columns=features)
        df.to_csv(os.path.join(ROOT_DIR, "results", feature_set, f'loss-{model_name}.csv'), index=False)
        plt.figure(figsize=(30, 16))
        plt.suptitle(f'loss - {title}', fontsize=30)
        for i in range(feature_size):
            plt.subplot(math.ceil(feature_size / 2), 2, i + 1)
            feature_loss = losses[:, i]
            plt.hist(x=feature_loss, bins='auto')
            plt.grid(axis='y')
            plt.title(f'{features[i]}')
        for i in range(1, 10):
            file_name = f'loss - {title}.png'
            if not os.path.exists(os.path.join(ROOT_DIR, "results", feature_set, file_name)):
                plt.savefig(os.path.join(ROOT_DIR, "results", feature_set, file_name))
                # plt.show()
                break

    plt.figure(figsize=(30, 16))
    plt.suptitle(title, fontsize=30)
    plot_length = min(1000, ((inputs_test.shape[0] - window_length - jump) // 100) * 100)
    x = np.arange(plot_length).reshape(-1, 1)

    for i in range(feature_size):
        plt.subplot(math.ceil(feature_size / 2), 2, i + 1)
        plt.plot(x, np.array(outputs[:plot_length, i]).reshape(-1, 1), label='predict', c='r', marker='.')
        plt.plot(x, np.array(targets_test[:plot_length, 0, i]).reshape(-1, 1), label='target', c='b', marker='.')
        plt.title(f'{features[i]}')

    for i in range(1, 10):
        file_name = f'{title}.png'
        if not os.path.exists(os.path.join(ROOT_DIR, "results", feature_set, file_name)):
            plt.savefig(os.path.join(ROOT_DIR, "results", feature_set, file_name))
            # plt.show()
            break


def multi_time_serial_lstm_crude_attention(monitor_window_length, window_sample_interval, target_skip_steps, batch_size,
                                           epochs_num, feature_names):
    data_reader = DatasetReader(
        [
            "20181113_Driver1_Trip1.hdf", "20181113_Driver1_Trip2.hdf", "20181116_Driver1_Trip3.hdf",
            "20181116_Driver1_Trip4.hdf", "20181116_Driver1_Trip5.hdf", "20181116_Driver1_Trip6.hdf",
            "20181117_Driver1_Trip7.hdf", "20181117_Driver1_Trip8.hdf", "20181203_Driver1_Trip9.hdf",
            "20181203_Driver1_Trip10.hdf",
        ])
    # feature_names = []
    # feature_names = ['CAN/EngineSpeed_CAN', 'CAN/VehicleSpeed', '/Plugins/Velocity_X']
    inputs_train, targets_train, inputs_test, targets_test, feature_names, normalized_test_data_time_serials = data_reader.sample(
        feature_names,
        time_steps=monitor_window_length,
        sample_interval=window_sample_interval,
        target_sequence=False,
        split_test_from_train=True,
        target_skip_steps=target_skip_steps,
        test_files=["20181117_Driver1_Trip7.hdf"])
    print("train input shape", inputs_train.shape)
    print("train target shape", targets_train.shape)
    print("test input shape", inputs_test.shape)
    print("test target shape", targets_test.shape)

    feature_size = inputs_test.shape[2]

    inputs_train_diff_min = np.min(np.max(inputs_train, axis=1) - np.min(inputs_train, axis=1), axis=1)
    sample_weights = np.select(
        [inputs_train_diff_min < 0.05, inputs_train_diff_min < 0.1, inputs_train_diff_min >= 0.1], [1, 3, 6])

    # prin_input(inputs_train, targets_train, sample_weights, feature_size, monitor_window_length, target_skip_steps, rows=8, save=False, show=True)

    model = keras.Sequential()
    model.add(layers.Bidirectional(layers.LSTM(128, dropout=0.05, return_sequences=False)))
    model.add(TemporalCrudeAttention(return_sequences=True))
    model.add(layers.Dense(32))
    model.add(layers.Dense(feature_size))
    model.build(input_shape=(batch_size, monitor_window_length, feature_size))
    model.compile(optimizer='adam', loss=loss_fn)

    # model.summary()
    # exit(0)

    model.fit(inputs_train, targets_train, validation_split=0.2, batch_size=batch_size, epochs=epochs_num, shuffle=True,
              sample_weight=sample_weights)

    res = model.evaluate(inputs_test, targets_test)
    print(res)

    plot_length = min(2000, ((inputs_test.shape[0] - monitor_window_length - target_skip_steps) // 100) * 100)

    outputs = []
    for i in range(0, inputs_test.shape[0], 100):
        outputs.append(model(inputs_test[i:i + 100, :, :]))
    outputs = np.vstack(outputs)
    x = np.arange(outputs.shape[0]).reshape(-1, 1)

    plt.figure(figsize=(30, 16))
    plt.suptitle(
        f'{feature_size} Features LSTM Crude Attn - {monitor_window_length} TimeSteps - Jump {target_skip_steps} Steps - {window_sample_interval}StepInterval-{batch_size}Batch-{epochs_num}Epochs')

    rows = min(4, feature_size)
    for i in range(feature_size):
        plt.subplot(rows, math.ceil(feature_size / rows), i + 1)
        plt.plot(x, np.array(outputs[:, i]).reshape(-1, 1), label='predict', c='r', marker='.')
        plt.plot(x, targets_test[:plot_length, 0, i], label='target', c='b', marker='.')
        plt.title(f'{feature_names[i]}')

    # sorted_outputs = np.array(list(map(list, zip(*flat_results[:plot_length])))[0])
    # sorted_targets = np.array(list(map(list, zip(*flat_results[:plot_length])))[1])
    # for i in range(feature_size):
    #     plt.subplot(4, math.ceil(feature_size / 2), feature_size + i + 1)
    #     plt.scatter(x, sorted_outputs[:, i].reshape(1, -1), label='predict', c='r', marker='+')
    #     plt.scatter(x, sorted_targets[:, 0, i].reshape(1, -1), label='target', c='b', marker='x')
    #     plt.title(f'{feature_names[i]}')
    for i in range(1, 10):
        file_name = f"mul-lstm-crude-attention-res{res}-{i}.png"
        if not os.path.exists(os.path.join(ROOT_DIR, "results", file_name)):
            plt.savefig(os.path.join(ROOT_DIR, "results", file_name))
            # plt.show()
            break


def sample_from_np(np_inputs, window_length=50):
    input_time_serials = []
    size = np_inputs.shape[0]
    for i in range(size - window_length):
        a_sample = np_inputs[i:i + window_length]
        input_time_serials.append(a_sample)
    return np.array(input_time_serials)


def define_threshold(df):
    df = pd.DataFrame(np.sort(df.values, axis=0), index=df.index, columns=df.columns)
    size = df.shape[0]
    percents = np.arange(5, 10) * math.floor(0.1 * size)
    thresholds = df.iloc[percents]
    return thresholds


def compare_with_threshold(outputs, thresholds, window_len=50):
    results = []
    for threshold in thresholds:
        # result = np.less_equal(outputs, threshold)
        # zeros = np.zeros([ window_len, result.shape[1]], dtype=bool)
        # result = np.vstack((zeros, result))  # [threshold_size, time_serial, feature_size]
        results.append(np.less_equal(outputs, threshold))
    results = np.array(results)
    return results


def results_to_span(results):
    end = results.shape[1]
    spans = []

    for j in range(results.shape[2]):  # features
        spans.append([])
        for i in range(results.shape[0]):
            start, stop = 0, 0
            spans[j].append([])
            while start < end:
                while start < end and results[i, start, j]:
                    start += 1
                stop = start
                while stop < end and not results[i, stop, j]:
                    stop += 1
                if stop > start:
                    spans[j][i].append([start, stop])
                start = stop

    return spans


def load_model_weights(window_length=50, jump=0, batch_size=64,
                       feature_set='selected_features', features=None, cell_num=128, dense_dim=64,
                       bidirection=False, attention=True, attn_layer=1):
    feature_size = len(features)
    model = keras.Sequential()
    # model.add(layers.LSTM(cell_num, dropout=0.05, return_sequences=False))
    a_layer = layers.LSTM(cell_num, dropout=0.05, return_sequences=False)
    if bidirection:
        a_layer = layers.Bidirectional(a_layer)
    if attention:
        a_layer = MultiHead(a_layer, layer_num=attn_layer)
    model.add(a_layer)
    if attention:
        model.add(layers.Flatten())
    model.add(layers.Dense(dense_dim))
    model.add(layers.Dense(feature_size))
    model.build(input_shape=(batch_size, window_length, feature_size))
    model.compile(optimizer='adam', loss=loss_fn)

    model_name = f'{cell_num}cell-{"Bi" if bidirection else "Uni"}-LSTM-{f"{attn_layer}lyrs-Attn-" if attention else ""}wl{window_length}-jp{jump}-{dense_dim}'
    model_weight_path = os.path.join(ROOT_DIR, 'models_weights', feature_set, f'{model_name}/checkpoint')
    if os.path.exists(model_weight_path):
        print("Load Model")
        model.load_weights(model_weight_path)
    return model, model_name


def detect_anomalies(train_files, test_file, feature_set, feature_names, window_length=50, jump=0, batch_size=64,
                     cell_num=128, dense_dim=64,
                     bidirection=True, attention=True, attn_layer=4):
    data_reader = DatasetReader(train_files + [test_file])
    scalers = data_reader.get_scalers(file_names=train_files, feature_names=feature_names)

    time_serials, _ = data_reader._concatenate_data(file_names=[test_file], feature_names=feature_names)

    anomaly_serials, anomalies = insert_super_anomalies(time_serials, 100, 50)
    while len(anomalies) == 0:
        anomaly_serials, anomalies = insert_super_anomalies(time_serials, 100, 50)
    print(f'anomaly_serials shape: {anomaly_serials.shape}')
    print('anomalies:')
    print(anomalies)

    normalized_time_serials = []
    for i in range(len(scalers)):
        scaler = scalers[i]
        normalized_time_serials.append(scaler.transform(time_serials[:, i].reshape([-1, 1])))
    normalized_time_serials = np.hstack(normalized_time_serials)
    print(f'normalized_time_serials shape: {normalized_time_serials.shape}')

    normalized_anomaly_serials = []
    for i in range(len(scalers)):
        scaler = scalers[i]
        normalized_anomaly_serials.append(scaler.transform(anomaly_serials[:, i].reshape([-1, 1])))
    normalized_anomaly_serials = np.hstack(normalized_anomaly_serials)
    print(f'normalized_anomaly_serials shape: {normalized_anomaly_serials.shape}')

    anomaly_input_samples = sample_from_np(normalized_anomaly_serials)
    input_samples = sample_from_np(normalized_time_serials)
    print(f'inputs shape: {anomaly_input_samples.shape}')

    model, model_name = load_model_weights(window_length=window_length, jump=jump, batch_size=batch_size,
                                           feature_set=feature_set, features=feature_names, cell_num=cell_num,
                                           dense_dim=dense_dim,
                                           bidirection=bidirection, attention=attention, attn_layer=attn_layer)

    threshold_csv_path = os.path.join(ROOT_DIR, 'results', feature_set, f'loss-{model_name}.csv')
    df = pd.read_csv(threshold_csv_path)
    thresholds = define_threshold(df)
    thresholds = thresholds.values

    outputs = []
    for i in range(0, anomaly_input_samples.shape[0], 100):
        outputs.append(model(anomaly_input_samples[i:i + 100, :, :]))
    outputs = np.vstack(outputs)
    outputs = np.vstack((normalized_anomaly_serials[:50], outputs))
    print(f'outputs shape: {outputs.shape}')

    losses = np.abs(normalized_anomaly_serials - outputs)
    print(f'losses shape: {outputs.shape}')

    # print(losses[:50, 0])
    # exit(0)

    # todo: compare??
    results = compare_with_threshold(losses, thresholds)
    print(f'results shape: {results.shape}')
    # print(np.all(results==False))
    spans = results_to_span(results)  # shape: [feature_size, threshold_size, ...]

    print_all(normal_serials=normalized_time_serials, anomaly_serials=normalized_anomaly_serials, anomalies=anomalies,
              outputs=outputs, feature_names=feature_names, spans=spans,
              path=os.path.join(ROOT_DIR, "results", feature_set, 'attack'))


data_dir = os.path.join(ROOT_DIR, "..", "data")
test_files = [f for f in listdir(data_dir) if isfile(os.path.join(data_dir, f)) and f[-3:] == "hdf" and f[15] != "1"]

selected_features = ['/CAN/Yawrate1', '/CAN/ENG_Trq_DMD', '/CAN/VehicleSpeed', '/CAN/AccPedal', '/CAN/ENG_Trq_ZWR',
                     '/Plugins/Velocity_X', '/GPS/Direction', '/CAN/EngineTemperature']
speed_features = ['/CAN/AccPedal', '/CAN/EngineSpeed_CAN', '/CAN/VehicleSpeed', '/GPS/Velocity',
                  '/Plugins/Accelerometer_X', '/Plugins/Body_acceleration_X', '/Plugins/Velocity_X',
                  '/CAN/WheelSpeed_RL']
corelated_features = ['/CAN/AccPedal', '/CAN/ENG_Trq_DMD', '/CAN/ENG_Trq_ZWR', '/CAN/ENG_Trq_m_ex',
                      '/CAN/EngineSpeed_CAN', '/CAN/OilTemperature1', '/CAN/Trq_Indicated', '/CAN/VehicleSpeed',
                      '/CAN/WheelSpeed_FL', '/CAN/WheelSpeed_FR', '/CAN/WheelSpeed_RL', '/CAN/WheelSpeed_RR']

sub_corelated_features = ['/CAN/AccPedal', '/CAN/ENG_Trq_ZWR', '/CAN/ENG_Trq_m_ex',
                      '/CAN/EngineSpeed_CAN' '/CAN/Trq_Indicated', '/CAN/VehicleSpeed',
                      '/CAN/WheelSpeed_FL', '/CAN/WheelSpeed_FR', '/CAN/WheelSpeed_RL', '/CAN/WheelSpeed_RR']

# # Train bare LSTM: LSTM with 128 cells dense to 64 then feature_num
# lstm(window_length=50, sample_interval=10, jump=0, batch_size=64, epochs_num=1,
#      feature_set='selected_features', features=selected_features, cell_num=128, dense_dim=64,
#      bidirection=False, attention=False, attn_layer=0,
#      test_only=False, test_file="20181117_Driver1_Trip7.hdf", plot_loss=True)

# # Train LSTM with one head attn: LSTM with 128 cells, 1 layer attn, dense to 64 then feature_num
# lstm(window_length=50, sample_interval=10, jump=0, batch_size=64, epochs_num=1,
#      feature_set='selected_features', features=selected_features, cell_num=128, dense_dim=64,
#      bidirection=False, attention=True, attn_layer=1,
#      test_only=False, test_file="20181117_Driver1_Trip7.hdf", plot_loss=True)
#
# # Train bare bi-direction LSTM: bi-direction LSTM with 128 cells, dense to 64 then feature_num
# lstm(window_length=50, sample_interval=10, jump=0, batch_size=64, epochs_num=1,
#      feature_set='selected_features', features=selected_features, cell_num=128, dense_dim=64,
#      bidirection=True, attention=False, attn_layer=0,
#      test_only=False, test_file="20181117_Driver1_Trip7.hdf", plot_loss=True)
#
# # Train bi-direction LSTM with one head attn: bi-direction LSTM with 128 cells, 1 layer attn, dense to 64 then feature_num
# lstm(window_length=50, sample_interval=10, jump=0, batch_size=64, epochs_num=1,
#      feature_set='selected_features', features=selected_features, cell_num=128, dense_dim=64,
#      bidirection=True, attention=True, attn_layer=1,
#      test_only=False, test_file="20181117_Driver1_Trip7.hdf", plot_loss=True)
#
# # Train bi-direction LSTM with one head attn: bi-direction LSTM with 128 cells, 2 layer attn, dense to 64 then feature_num
# lstm(window_length=50, sample_interval=10, jump=0, batch_size=64, epochs_num=1,
#      feature_set='selected_features', features=selected_features, cell_num=128, dense_dim=64,
#      bidirection=True, attention=True, attn_layer=2,
#      test_only=False, test_file="20181117_Driver1_Trip7.hdf", plot_loss=True)
#
# Train bi-direction LSTM with one head attn: bi-direction LSTM with 128 cells, 4 layer attn, dense to 64 then feature_num
lstm(window_length=50, sample_interval=10, jump=0, batch_size=64, epochs_num=80,
     feature_set='sub_corelated_features', features=sub_corelated_features, cell_num=128, dense_dim=64,
     bidirection=True, attention=True, attn_layer=4,
     test_only=False, test_file="20181117_Driver1_Trip7.hdf", plot_loss=True)

detect_anomalies(train_files=train_files, test_file="20181117_Driver1_Trip7.hdf", feature_set='sub_corelated_features',
                 feature_names=sub_corelated_features, window_length=50, jump=0, batch_size=64,
                 cell_num=128, dense_dim=64, bidirection=True, attention=True, attn_layer=4)

# for test_file in test_files:
#     model_test(train_files=train_files, test_file=test_file, feature_names=selected_features, layer_num=12, dense_1_num=256, dense_2_num=64, monitor_window_length=200, target_skip_steps=4)
# for test_file in test_files:
#     model_test(train_files=train_files, test_file=test_file, feature_names=selected_features, layer_num=16,
#                dense_1_num=256, dense_2_num=64, monitor_window_length=200, target_skip_steps=4)
