import math
import random
import os
from os import listdir
from os.path import isfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from classic_rnn.TemporalCrudeAttention import TemporalCrudeAttention

from tools.DatasetReader import DatasetReader
from keras_multi_head import MultiHead

from tools.anomaly_creator import insert_super_anomalies, print_all, sample_from_np, detect_results_to_spans, \
    compare_with_threshold, compare_with_threshold_max, span_analyze

tf.get_logger().setLevel('ERROR')

tf.keras.backend.set_floatx('float64')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_files = [
    "20181113_Driver1_Trip1.hdf", "20181113_Driver1_Trip2.hdf", "20181116_Driver1_Trip3.hdf",
    "20181116_Driver1_Trip4.hdf", "20181116_Driver1_Trip5.hdf", "20181116_Driver1_Trip6.hdf",
    "20181117_Driver1_Trip8.hdf", "20181117_Driver1_Trip7.hdf"
]
# test_file = "20181117_Driver1_Trip7.hdf"
vali_file = "20181203_Driver1_Trip9.hdf"
attack_file = "20181203_Driver1_Trip10.hdf"
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


sample_interval = 4
jump = 0
batch_size = 64

feature_set = 'sub_corelated_features'
feature_names = sub_corelated_features

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
    # return tf.reduce_mean(tf.abs(tf.subtract(tf.reshape(y_true, [-1, y_true.shape[2]]), y_pred)), axis=1)
    # return tf.reduce_mean(tf.abs(tf.subtract(tf.math.log(tf.reshape(y_true+1, [-1, y_true.shape[2]])), tf.math.log(y_pred+1))), axis=1)
    # y_true = tf.reshape(y_true, [-1, y_true.shape[2]])
    return tf.keras.metrics.mean_squared_logarithmic_error(y_true, y_pred)

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

def train_model(window_length=50, epochs_num=10, cell_num=128, dense_dim=64,
         bidirection=False, attention=False, attn_layer=4, continue_epoch=-1):

    data_reader = DatasetReader(train_files + [vali_file])
    inputs_train, targets_train, inputs_test, targets_test, features, _ = data_reader.sample(
        feature_names=feature_names,
        time_steps=window_length,
        sample_interval=sample_interval,
        target_sequence=False,
        split_test_from_train=False,
        target_skip_steps=jump,
        test_files=[vali_file])
    targets_train = np.reshape(targets_train, [-1, targets_train.shape[2]])
    targets_test = np.reshape(targets_test, [-1, targets_test.shape[2]])

    print(f'inputs_train:{inputs_train.shape},targets_train:{targets_train.shape},inputs_test:{inputs_test.shape},targets_test:{targets_test.shape}')
    feature_size = len(feature_names)

    model, model_name = load_model_weights(window_length=window_length, jump=jump, batch_size=batch_size,
                                           feature_set=feature_set, features=feature_names, cell_num=cell_num,
                                           dense_dim=dense_dim, bidirection=bidirection, attention=attention,
                                           attn_layer=attn_layer, epoch=continue_epoch)

    model_weight_root_path = os.path.join(ROOT_DIR, 'models_weights', feature_set, f'{model_name}')

    history_csv_path = os.path.join(ROOT_DIR, "results", feature_set, f'history-{model_name}.csv')
    records = []
    if epochs_num > 0:
        evaluation_res = []
        train_res = []
        validate_res = []
        for i in range(epochs_num):
            h = model.fit(inputs_train, targets_train, batch_size=batch_size, epochs=1,
                          shuffle=True, verbose=0)
            eval_loss = model.evaluate(inputs_test, targets_test, verbose=0)
            train_loss = h.history['loss'][-1]
            # vali_loss = h.history['val_loss'][-1]
            evaluation_res.append(eval_loss)
            train_res.append(train_loss)
            # validate_res.append(vali_loss)
            print(f"{i} - train_loss:{train_loss},eval_loss:{eval_loss}")
            model.save_weights(os.path.join(model_weight_root_path, f'epoch{continue_epoch + i + 1}/checkpoint'))

        train_res = np.array(train_res).reshape([-1, 1])
        validate_res = np.array(validate_res).reshape([-1, 1])
        evaluation_res = np.array(evaluation_res).reshape([-1, 1])
        records = pd.DataFrame(np.hstack([train_res, evaluation_res]),
                               columns=['train_loss', 'eval_loss'])

        if os.path.exists(history_csv_path):
            history = pd.read_csv(history_csv_path)
            records = pd.concat([history, records], ignore_index=True)
        print(f'loss records of {model_name}')
        records.to_csv(history_csv_path, index=False)

        best_epoch = find_lowest_loss_epoch(history_csv_path)
        print(f'best_epoch: {best_epoch}')

        model, model_name = load_model_weights(window_length=window_length, jump=jump, batch_size=batch_size,
                                               feature_set=feature_set, features=feature_names, cell_num=cell_num,
                                               dense_dim=dense_dim, bidirection=bidirection, attention=attention,
                                               attn_layer=attn_layer, epoch=best_epoch)

        # print_train_loss(inputs_train, targets_train, model, model_name)
        print_train_loss(inputs_test, targets_test, model, model_name)

    outputs = []
    for i in range(0, inputs_test.shape[0], 100):
        outputs.append(model(inputs_test[i:i + 100, :, :]))
    outputs = np.vstack(outputs)
    plt.figure(figsize=(30, 16))
    # plt.suptitle(title, fontsize=30)
    plot_length = min(1000, ((inputs_test.shape[0] - window_length - jump) // 100) * 100)
    x = np.arange(plot_length).reshape(-1, 1)

    title = f'{feature_size} {feature_set} {cell_num}cells {"Bi" if bidirection else "Uni"} LSTM {f"{attn_layer}lyrs Attn" if attention else ""} - {dense_dim} - {window_length}TS - Jump{jump}'

    for i in range(feature_size):
        plt.subplot(math.ceil(feature_size / 2), 2, i + 1)
        plt.plot(x, np.array(outputs[:plot_length, i]).reshape(-1, 1), label='predict', c='r', marker='.')
        plt.plot(x, np.array(targets_test[:plot_length, i]).reshape(-1, 1), label='target', c='b', marker='.')
        plt.title(f'{features[i]}')

    for i in range(1, 10):
        file_name = f'{title}-{i}.png'
        if not os.path.exists(os.path.join(ROOT_DIR, "results", feature_set, file_name)):
            plt.savefig(os.path.join(ROOT_DIR, "results", feature_set, file_name))
            # plt.show()
            break
    print('==========')

def find_lowest_loss_epoch(history_csv_path):
    history_df = pd.read_csv(history_csv_path)
    row_id_min_eval_loss = history_df['eval_loss'].idxmin()
    return row_id_min_eval_loss+1


def print_train_loss(inputs_train, targets_train, model, model_name):
    outputs = []

    for i in range(0, inputs_train.shape[0], 100):
        outputs.append(model(inputs_train[i:i + 100, :, :]))
    outputs = np.vstack(outputs)
    print(f'---output {outputs.shape}')

    losses = []
    for i in range(outputs.shape[0]):
        # losses.append(np.abs(outputs[i] - targets_train[i]))
        losses.append(np.square(outputs[i] - targets_train[i]))
        # losses.append(np.square(np.log(outputs[i] + 1) - np.log(targets_train[i] + 1)))
    losses = np.array(losses)
    # losses = tf.keras.metrics.mean_squared_logarithmic_error(outputs, targets_train).numpy().reshape([-1,1])
    print(f'---losses {losses.shape}')

    df = pd.DataFrame(data=losses, columns=feature_names)
    df.to_csv(os.path.join(ROOT_DIR, "results", feature_set, f'loss-{model_name}.csv'), index=False)
    # plt.figure(figsize=(30, 16))
    # plt.suptitle(f'loss - {title}', fontsize=30)
    # for i in range(feature_size):
    #     plt.subplot(math.ceil(feature_size / 2), 2, i + 1)
    #     feature_loss = losses[:, i]
    #     plt.hist(x=feature_loss, bins='auto')
    #     plt.grid(axis='y')
    #     plt.title(f'{features[i]}')
    # for i in range(1, 10):
    #     file_name = f'loss - {title}.png'
    #     if not os.path.exists(os.path.join(ROOT_DIR, "results", feature_set, file_name)):
    #         plt.savefig(os.path.join(ROOT_DIR, "results", feature_set, file_name))
    #         # plt.show()
    #         break


# def lstm(window_length=50, epochs_num=10, cell_num=128, dense_dim=64,
#          bidirection=False, attention=False, attn_layer=4,
#          test_only=False, plot_loss=True):
#
#     model_name = f'{cell_num}cell-{"Bi" if bidirection else "Uni"}-LSTM-{f"{attn_layer}lyrs-Attn-" if attention else ""}wl{window_length}-jp{jump}-{dense_dim}'
#     model_weight_root_path = os.path.join(ROOT_DIR, 'models_weights', feature_set, f'{model_name}')
#     print(f'model: {model_name}')
#
#     data_reader = DatasetReader(train_files + [test_file])
#
#     inputs_train, targets_train, inputs_test, targets_test, features, _ = data_reader.sample(feature_names=feature_names,
#                                                                                              time_steps=window_length,
#                                                                                              sample_interval=sample_interval,
#                                                                                              target_sequence=False,
#                                                                                              split_test_from_train=False,
#                                                                                              target_skip_steps=jump,
#                                                                                              test_files=[test_file])
#     targets_train = np.reshape(targets_train, [-1, targets_train.shape[2]])
#     targets_test = np.reshape(targets_test, [-1, targets_test.shape[2]])
#
#     print(f'inputs_train:{inputs_train.shape},targets_train:{targets_train.shape},inputs_test:{inputs_test.shape},targets_test:{targets_test.shape}')
#     feature_size = len(features_names)
#
#     # inputs_train_diff_min = np.min(np.max(inputs_train, axis=1) - np.min(inputs_train, axis=1), axis=1)
#     # sample_weights = np.select([inputs_train_diff_min < 0.05, inputs_train_diff_min < 0.1, inputs_train_diff_min >= 0.1], [1, 3, 6])
#
#     # prin_input(inputs_train, targets_train, sample_weights, feature_size, monitor_window_length, target_skip_steps, rows=8, save=False, show=True)
#
#     model = keras.Sequential()
#     # model.add(layers.LSTM(cell_num, dropout=0.05, return_sequences=False))
#     a_layer = layers.LSTM(cell_num, dropout=0.05, return_sequences=False)
#     if bidirection:
#         a_layer = layers.Bidirectional(a_layer)
#     if attention:
#         a_layer = MultiHead(a_layer, layer_num=attn_layer)
#     model.add(a_layer)
#     if attention:
#         model.add(layers.Flatten())
#     model.add(layers.Dense(dense_dim))
#     model.add(layers.Dense(feature_size))
#     model.build(input_shape=(batch_size, window_length, feature_size))
#     model.compile(optimizer='adam', loss=loss_fn)
#
#     # if os.path.exists(model_weight_path):
#     #     print(f"Load Model:{model_name}")
#     #     model.load_weights(model_weight_path)
#
#     # model.fit(inputs_train, targets_train, validation_split=0.2, batch_size=batch_size, epochs=epochs_num, shuffle=True, sample_weight=sample_weights)
#     if not test_only:
#         if epochs_num > 0:
#             evaluation_res = []
#             train_res = []
#             validate_res = []
#             for i in range(epochs_num):
#                 h = model.fit(inputs_train, targets_train, validation_split=0.2, batch_size=batch_size, epochs=1, shuffle=True, verbose=0)
#                 eval_loss = model.evaluate(inputs_test, targets_test, verbose=0)
#                 train_loss = h.history['loss'][-1]
#                 vali_loss = h.history['val_loss'][-1]
#                 evaluation_res.append(eval_loss)
#                 train_res.append(train_loss)
#                 validate_res.append(vali_loss)
#                 print(f"{i} - train_loss:{train_loss},val_loss:{vali_loss},eval_loss:{eval_loss}")
#
#                 model.save_weights(os.path.join(model_weight_root_path, f'epoch{i+1}/checkpoint'))
#
#             history_csv_path = os.path.join(ROOT_DIR, "results", feature_set, f'history-{model_name}.csv')
#             train_res = np.array(train_res).reshape([-1,1])
#             validate_res = np.array(validate_res).reshape([-1,1])
#             evaluation_res = np.array(evaluation_res).reshape([-1,1])
#             records = pd.DataFrame(np.hstack([train_res, validate_res, evaluation_res]), columns=['train_loss', 'val_loss', 'eval_loss'])
#
#             if os.path.exists(history_csv_path):
#                 history = pd.read_csv(history_csv_path)
#                 records = pd.concat([history, records], ignore_index=True)
#             print(f'loss records of {model_name}')
#             records.to_csv(history_csv_path, index=False)
#
#     res = model.evaluate(inputs_test, targets_test, verbose=0)
#     print(f'evaluate result: {res}')
#
#     title = f'{feature_size} {feature_set} {cell_num}cells {"Bi" if bidirection else "Uni"} LSTM {f"{attn_layer}lyrs Attn" if attention else ""} - {dense_dim} - {window_length}TS - Jump{jump} - Res-{res:.5f} - {test_file[15:-4]}'
#
#     if plot_loss:
#         outputs = []
#         losses = []
#         for i in range(inputs_train.shape[0]):
#             input = inputs_train[i, :, :]
#             target = targets_train[i, :, :]
#             output = model(input)
#             outputs.append(output)
#             loss = tf.keras.metrics.mean_squared_logarithmic_error(output, target)
#             losses.append(loss)
#         # outputs = np.array(outputs)
#         losses = np.array(losses)
#
#         df = pd.DataFrame(data=losses, columns=features)
#         df.to_csv(os.path.join(ROOT_DIR, "results", feature_set, f'loss-{model_name}.csv'), index=False)
#         plt.figure(figsize=(30, 16))
#         plt.suptitle(f'loss - {title}', fontsize=30)
#         for i in range(feature_size):
#             plt.subplot(math.ceil(feature_size / 2), 2, i + 1)
#             feature_loss = losses[:, i]
#             plt.hist(x=feature_loss, bins='auto')
#             plt.grid(axis='y')
#             plt.title(f'{features[i]}')
#         for i in range(1, 10):
#             file_name = f'loss - {title}.png'
#             if not os.path.exists(os.path.join(ROOT_DIR, "results", feature_set, file_name)):
#                 plt.savefig(os.path.join(ROOT_DIR, "results", feature_set, file_name))
#                 # plt.show()
#                 break
#
#     outputs = []
#     for i in range(0, inputs_test.shape[0], 100):
#         outputs.append(model(inputs_test[i:i + 100, :, :]))
#     outputs = np.vstack(outputs)
#
#     plt.figure(figsize=(30, 16))
#     plt.suptitle(title, fontsize=30)
#     plot_length = min(1000, ((inputs_test.shape[0] - window_length - jump) // 100) * 100)
#     x = np.arange(plot_length).reshape(-1, 1)
#
#     for i in range(feature_size):
#         plt.subplot(math.ceil(feature_size / 2), 2, i + 1)
#         plt.plot(x, np.array(outputs[:plot_length, i]).reshape(-1, 1), label='predict', c='r', marker='.')
#         plt.plot(x, np.array(targets_test[:plot_length, i]).reshape(-1, 1), label='target', c='b', marker='.')
#         plt.title(f'{features[i]}')
#
#     for i in range(1, 10):
#         file_name = f'{title}.png'
#         if not os.path.exists(os.path.join(ROOT_DIR, "results", feature_set, file_name)):
#             plt.savefig(os.path.join(ROOT_DIR, "results", feature_set, file_name))
#             # plt.show()
#             break
#     print('==========')


def define_threshold(df):
    df = pd.DataFrame(np.sort(df.values, axis=0), index=df.index, columns=df.columns)
    # size = df.shape[0]
    # percents = np.arange(9, 11) * math.floor(0.1 * size)
    # thresholds = df.iloc[percents].to_numpy()
    end = df.iloc[-1].to_numpy()
    # thresholds = np.vstack((thresholds, np.array([end*1.1, end*1.2, end*1.3])))
    thresholds = np.array([end*0.6, end*0.8, end, end*1.2, end*1.4])
    return thresholds


def load_model_weights(window_length=50, jump=0, batch_size=64,
                       feature_set='selected_features', features=None, cell_num=128, dense_dim=64,
                       bidirection=False, attention=True, attn_layer=1, epoch=1):
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
    model.compile(optimizer='adam', loss='mse')
    # model.compile(optimizer='adam', loss=loss_fn)
    # model.summary()

    model_name = f'{cell_num}cell-{"Bi" if bidirection else "Uni"}-LSTM-{f"{attn_layer}lyrs-Attn-" if attention else ""}wl{window_length}-jp{jump}-{dense_dim}'
    if epoch > 0:
        print(f'continue from {epoch}')
        model_weight_path = os.path.join(ROOT_DIR, 'models_weights', feature_set, f'{model_name}', f'epoch{epoch}/checkpoint')
        if os.path.exists(model_weight_path):
            print("Load Model")
            model.load_weights(model_weight_path)
    return model, model_name

def plot_detect_res(file_path):
    df = pd.read_csv(file_path)
    model_names = df.iloc[:, 0].to_list()
    threshold_TP_rates = np.reshape(df.iloc[:, 1:6].to_numpy(), [-1,5])
    threshold_TP_pers = np.reshape(df.iloc[:, 6:11].to_numpy(), [-1,5])
    threshold_TP_avg_delays = np.reshape(df.iloc[:, 11:16].to_numpy(), [-1,5])
    threshold_FP_pers = np.reshape(df.iloc[:, 16:21].to_numpy(), [-1,5])

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    threshold_size = len(threshold_TP_rates[0])
    model_size = len(model_names)
    x = np.arange(threshold_size)

    # for i in range(threshold_size):
    #     a_model_TP_rates = threshold_TP_rates[:, i]
    #     ax[0].plot(x, a_model_TP_rates, legend=)

    for i in range(model_size):
        a_model_TP_rates = threshold_TP_rates[i]
        ax[0].plot(x, a_model_TP_rates, label=model_names[i])

        a_threshold_TP_avg_delay = threshold_TP_avg_delays[i]
        ax[1].plot(x, a_threshold_TP_avg_delay, label=model_names[i])

        threshold_TP_per = threshold_TP_pers[i]
        ax[2].plot(x, threshold_TP_per, label=model_names[i])

        a_threshold_FP_per = threshold_FP_pers[i]
        ax[3].plot(x, a_threshold_FP_per, label=model_names[i])

    plt.legend()
    # plt.yscale('log')
    plt.show()
    # dim = 3
    # w = 0.75
    # dimw = w / dim
    # ax[0].bar(x, threshold_TP_rate, )

def write_loss(window_length=50, cell_num=128, dense_dim=64,
               bidirection=False, attention=False, attn_layer=4):
    data_reader = DatasetReader(train_files + [vali_file])
    inputs_train, targets_train, inputs_test, targets_test, features, _ = data_reader.sample(
        feature_names=feature_names,
        time_steps=window_length,
        sample_interval=sample_interval,
        target_sequence=False,
        split_test_from_train=False,
        target_skip_steps=jump,
        test_files=[vali_file])
    targets_train = np.reshape(targets_train, [-1, targets_train.shape[2]])
    targets_test = np.reshape(targets_test, [-1, targets_test.shape[2]])

    print(
        f'inputs_train:{inputs_train.shape},targets_train:{targets_train.shape},inputs_test:{inputs_test.shape},targets_test:{targets_test.shape}')
    model_name = f'{cell_num}cell-{"Bi" if bidirection else "Uni"}-LSTM-{f"{attn_layer}lyrs-Attn-" if attention else ""}wl{window_length}-jp{jump}-{dense_dim}'
    history_csv_path = os.path.join(ROOT_DIR, "results", feature_set, f'history-{model_name}.csv')
    best_epoch = find_lowest_loss_epoch(history_csv_path)
    model, model_name = load_model_weights(window_length=window_length, jump=jump, batch_size=batch_size,
                                  feature_set=feature_set, features=feature_names, cell_num=cell_num,
                                  dense_dim=dense_dim,
                                  bidirection=bidirection, attention=attention, attn_layer=attn_layer, epoch=best_epoch)

    print_train_loss(inputs_test, targets_test, model, model_name)


def detect_anomalies(window_length=50, cell_num=128, dense_dim=64,
                     bidirection=True, attention=True, attn_layer=4, attack_ids=None, avg_wl=3):
    data_reader = DatasetReader(train_files + [attack_file])
    scalers = data_reader.get_scalers(file_names=train_files, feature_names=feature_names)

    time_serials, _ = data_reader._concatenate_data(file_names=[attack_file], feature_names=feature_names)

    # anomaly_serials, anomalies = insert_super_anomalies(time_serials, feature_ids=attack_ids, max_anom_duration=100, cooldown=100, window_len=window_length, avg_anomaly_interval=400)
    # print(f'anomaly_serials shape: {anomaly_serials.shape}')
    # print('anomalies:')
    # print(anomalies)

    serials_df = pd.read_csv(os.path.join(ROOT_DIR, "results", 'sub_corelated_features', '20181203_Driver1_Trip10_Anomalous_Serials.csv'))
    anomaly_serials = serials_df.to_numpy(dtype=float)
    anomalies_df = pd.read_csv(os.path.join(ROOT_DIR, "results", 'sub_corelated_features', '20181203_Driver1_Trip10_Anomalies.csv'))
    anomalies = anomalies_df.values.tolist()

    normalized_time_serials = []
    normalized_anomaly_serials = []
    for i in range(len(scalers)):
        scaler = scalers[i]
        normalized_time_serials.append(scaler.transform(time_serials[:, i].reshape([-1, 1])))
        normalized_anomaly_serials.append(scaler.transform(anomaly_serials[:, i].reshape([-1, 1])))
    normalized_time_serials = np.hstack(normalized_time_serials)
    print(f'normalized_time_serials shape: {normalized_time_serials.shape}')
    normalized_anomaly_serials = np.hstack(normalized_anomaly_serials)
    print(f'normalized_anomaly_serials shape: {normalized_anomaly_serials.shape}')

    anomaly_input_samples = sample_from_np(normalized_anomaly_serials, window_length=window_length)
    print(f'inputs shape: {anomaly_input_samples.shape}')

    model_name = f'{cell_num}cell-{"Bi" if bidirection else "Uni"}-LSTM-{f"{attn_layer}lyrs-Attn-" if attention else ""}wl{window_length}-jp{jump}-{dense_dim}'
    history_csv_path = os.path.join(ROOT_DIR, "results", feature_set, f'history-{model_name}.csv')
    best_epoch = find_lowest_loss_epoch(history_csv_path)
    print(f'best_epoch: {best_epoch}')
    model, _ = load_model_weights(window_length=window_length, jump=jump, batch_size=batch_size,
                                           feature_set=feature_set, features=feature_names, cell_num=cell_num,
                                           dense_dim=dense_dim,
                                           bidirection=bidirection, attention=attention, attn_layer=attn_layer, epoch=best_epoch)
    threshold_csv_path = os.path.join(ROOT_DIR, 'results', feature_set, f'loss-{model_name}.csv')
    df = pd.read_csv(threshold_csv_path)
    thresholds = define_threshold(df)
    # print(f'thresholds: {thresholds}')

    outputs = []
    for i in range(0, anomaly_input_samples.shape[0], 100):
        outputs.append(model(anomaly_input_samples[i:i + 100, :, :]))
    outputs = np.vstack(outputs)
    print(f'outputs shape: {outputs.shape}')

    padded_outs = np.vstack((normalized_time_serials[:window_length+jump+1],outputs))
    losses = []
    for i in range(outputs.shape[0]):
        # losses.append(np.abs(padded_outs[i] - normalized_anomaly_serials[i]))
        losses.append(np.square(padded_outs[i] - normalized_anomaly_serials[i]))
    losses = np.array(losses)
    # losses = np.abs(normalized_anomaly_serials - padded_outs)
    # losses = tf.keras.metrics.mean_squared_logarithmic_error(padded_outs, normalized_time_serials).numpy().reshape([-1,1])

    print(f'losses shape: {losses.shape}')

    losses_copy = np.copy(losses)
    avg_losses = []
    for i in range(losses_copy.shape[1]):
        a_loss = losses_copy[:, i].reshape(-1)
        a_avg_loss = np.convolve(a_loss, np.ones(avg_wl) / avg_wl, mode='valid')
        avg_losses.append(a_avg_loss.reshape([-1, 1]))

    avg_losses = np.hstack(avg_losses)
    avg_losses = np.vstack((np.zeros([avg_wl-1, avg_losses.shape[1]]).astype(bool), avg_losses))
    print(avg_losses.shape)

    # results = compare_with_threshold(losses, thresholds)
    # results = compare_with_threshold_max(avg_losses, thresholds)
    # results = compare_with_threshold_max(losses, thresholds)
    results = compare_with_threshold(avg_losses, thresholds)
    print(f'results shape: {results.shape}')

    spans = detect_results_to_spans(results)  # shape: [feature_size, threshold_size, ...]
    print(spans)

    print(f'normalized_time_serials shape: {normalized_time_serials.shape}')
    print(f'normalized_anomaly_serials shape: {normalized_anomaly_serials.shape}')

    threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per, threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = span_analyze(spans, anomalies, normalized_time_serials.shape[0])

    path = os.path.join(ROOT_DIR, "results", feature_set, f'attack-{model_name}')
    os.makedirs(path, exist_ok=True)
    print_all(normal_serials=normalized_time_serials, anomaly_serials=normalized_anomaly_serials, anomalies=anomalies,
              outputs=padded_outs, feature_names=feature_names, spans=spans,
              path=path)


    return f'{"Bi" if bidirection else "Uni"} {f"{attn_layer}lyr" if attention else "No"}-Attn', threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
           threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration


# data_dir = os.path.join(ROOT_DIR, "..", "data")
# test_files = [f for f in listdir(data_dir) if isfile(os.path.join(data_dir, f)) and f[-3:] == "hdf" and f[15] != "1"]


epochs_num = 40
# feature_set = 'sub_corelated_features'
# features = sub_corelated_features
# batch_size = 128
# plot_loss = True

# # ######## determine cell number ########
# train_model(window_length=60, epochs_num=epochs_num,
#      cell_num=16, dense_dim=64,
#      bidirection=False, attention=False, attn_layer=1, continue_epoch=-1)
# train_model(window_length=60, epochs_num=epochs_num,
#      cell_num=32, dense_dim=64,
#      bidirection=False, attention=False, attn_layer=1, continue_epoch=-1)
# train_model(window_length=60, epochs_num=epochs_num,
#      cell_num=48, dense_dim=64,
#      bidirection=False, attention=False, attn_layer=1, continue_epoch=-1)
# train_model(window_length=60, epochs_num=epochs_num,
#      cell_num=64, dense_dim=64,
#      bidirection=False, attention=False, attn_layer=1, continue_epoch=-1)
# train_model(window_length=60, epochs_num=epochs_num,
#      cell_num=80, dense_dim=64,
#      bidirection=False, attention=False, attn_layer=1, continue_epoch=-1)
# train_model(window_length=60, epochs_num=epochs_num,
#      cell_num=96, dense_dim=64,
#      bidirection=False, attention=False, attn_layer=1, continue_epoch=-1)
# # ######## determine cell number ########
# exit(0)
best_cell_num_in_ws_50_l = 16
best_cell_num_in_ws_50 = 32
best_cell_num_in_ws_50_r = 48
epochs_num = 40

######## determine window length ######## Guess: cell number has relationship with window length
# train_model(window_length=30, epochs_num=epochs_num,
#      cell_num=best_cell_num_in_ws_50_l, dense_dim=64,
#      bidirection=False, attention=True, attn_layer=2, continue_epoch=-1)
# train_model(window_length=30, epochs_num=epochs_num,
#      cell_num=best_cell_num_in_ws_50, dense_dim=64,
#      bidirection=False, attention=True, attn_layer=2, continue_epoch=-1)
# train_model(window_length=30, epochs_num=epochs_num,
#      cell_num=best_cell_num_in_ws_50_r, dense_dim=64,
#      bidirection=False, attention=True, attn_layer=2, continue_epoch=-1)
# train_model(window_length=60, epochs_num=epochs_num,
#      cell_num=best_cell_num_in_ws_50_l, dense_dim=64,
#      bidirection=False, attention=True, attn_layer=2, continue_epoch=-1)
# train_model(window_length=60, epochs_num=epochs_num,
#      cell_num=best_cell_num_in_ws_50, dense_dim=64,
#      bidirection=False, attention=True, attn_layer=2, continue_epoch=-1)
# train_model(window_length=60, epochs_num=epochs_num,
#      cell_num=best_cell_num_in_ws_50_r, dense_dim=64,
#      bidirection=False, attention=True, attn_layer=2, continue_epoch=-1)
# train_model(window_length=90, epochs_num=epochs_num,
#      cell_num=best_cell_num_in_ws_50_l, dense_dim=64,
#      bidirection=False, attention=True, attn_layer=2, continue_epoch=-1)
# train_model(window_length=90, epochs_num=epochs_num,
#      cell_num=best_cell_num_in_ws_50, dense_dim=64,
#      bidirection=False, attention=True, attn_layer=2, continue_epoch=-1)
# train_model(window_length=90, epochs_num=epochs_num,
#      cell_num=best_cell_num_in_ws_50_r, dense_dim=64,
#      bidirection=False, attention=True, attn_layer=2, continue_epoch=-1)
####### determine window length ########

# todo: effectiveness of bi-lstm
# todo: summary of model

# exit(0)
best_cell_num = 16
best_window_length = 90

######## effectiveness of attn model ########
# lstm(window_length=best_window_length, sample_interval=2, batch_size=batch_size, epochs_num=epochs_num,
#      feature_set=feature_set, features=features, cell_num=best_cell_num, dense_dim=64,
#      bidirection=False, attention=False, attn_layer=1,
#      test_only=True, plot_loss=plot_loss)
# lstm(window_length=best_window_length, sample_interval=2, batch_size=batch_size, epochs_num=epochs_num,
#      feature_set=feature_set, features=features, cell_num=best_cell_num, dense_dim=64,
#      bidirection=True, attention=False, attn_layer=1,
#      test_only=True, plot_loss=plot_loss)
# lstm(window_length=best_window_length, sample_interval=2, batch_size=batch_size, epochs_num=epochs_num,
#      feature_set=feature_set, features=features, cell_num=best_cell_num, dense_dim=64,
#      bidirection=False, attention=True, attn_layer=1,
#      test_only=True, plot_loss=plot_loss)
# lstm(window_length=best_window_length, sample_interval=2, batch_size=batch_size, epochs_num=epochs_num,
#      feature_set=feature_set, features=features, cell_num=best_cell_num, dense_dim=64,
#      bidirection=True, attention=True, attn_layer=1,
#      test_only=True, plot_loss=plot_loss)
# lstm(window_length=best_window_length, sample_interval=2, batch_size=batch_size, epochs_num=epochs_num,
#      feature_set=feature_set, features=features, cell_num=best_cell_num, dense_dim=64,
#      bidirection=True, attention=True, attn_layer=2,
#      test_only=True, plot_loss=plot_loss)
# lstm(window_length=best_window_length, sample_interval=2, batch_size=batch_size, epochs_num=epochs_num,
#      feature_set=feature_set, features=features, cell_num=best_cell_num, dense_dim=64,
#      bidirection=True, attention=True, attn_layer=4,
#      test_only=True, plot_loss=plot_loss)
# lstm(window_length=best_window_length, sample_interval=2, batch_size=batch_size, epochs_num=epochs_num,
#      feature_set=feature_set, features=features, cell_num=best_cell_num, dense_dim=64,
#      bidirection=True, attention=True, attn_layer=8,
#      test_only=True, plot_loss=plot_loss)
# lstm(window_length=best_window_length, sample_interval=2, batch_size=batch_size, epochs_num=epochs_num,
#      feature_set=feature_set, features=features, cell_num=best_cell_num, dense_dim=64,
#      bidirection=True, attention=True, attn_layer=16,
#      test_only=True, plot_loss=plot_loss)

# ######## effectiveness of attn model ########
# write_loss(window_length=30, cell_num=16, dense_dim=64,
#          bidirection=False, attention=False, attn_layer=1)
# write_loss(window_length=30, cell_num=16, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=1)
# write_loss(window_length=30, cell_num=16, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=2)
# write_loss(window_length=60, cell_num=16, dense_dim=64,
#          bidirection=False, attention=False, attn_layer=1)
# write_loss(window_length=60, cell_num=16, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=1)
# write_loss(window_length=60, cell_num=16, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=2)
# write_loss(window_length=90, cell_num=16, dense_dim=64,
#          bidirection=False, attention=False, attn_layer=1)
# write_loss(window_length=90, cell_num=16, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=1)
# write_loss(window_length=90, cell_num=16, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=2)
# write_loss(window_length=30, cell_num=32, dense_dim=64,
#          bidirection=False, attention=False, attn_layer=1)
# write_loss(window_length=30, cell_num=32, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=1)
# write_loss(window_length=30, cell_num=32, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=2)
# write_loss(window_length=60, cell_num=32, dense_dim=64,
#          bidirection=False, attention=False, attn_layer=1)
# write_loss(window_length=60, cell_num=32, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=1)
# write_loss(window_length=60, cell_num=32, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=2)
# write_loss(window_length=90, cell_num=32, dense_dim=64,
#          bidirection=False, attention=False, attn_layer=1)
# write_loss(window_length=90, cell_num=32, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=1)
# write_loss(window_length=90, cell_num=32, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=2)
# write_loss(window_length=30, cell_num=48, dense_dim=64,
#          bidirection=False, attention=False, attn_layer=1)
# write_loss(window_length=30, cell_num=48, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=1)
# write_loss(window_length=30, cell_num=48, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=2)
# write_loss(window_length=60, cell_num=48, dense_dim=64,
#          bidirection=False, attention=False, attn_layer=1)
# write_loss(window_length=60, cell_num=48, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=1)
# write_loss(window_length=60, cell_num=48, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=2)
# write_loss(window_length=90, cell_num=48, dense_dim=64,
#          bidirection=False, attention=False, attn_layer=1)
# write_loss(window_length=90, cell_num=48, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=1)
# write_loss(window_length=90, cell_num=48, dense_dim=64,
#          bidirection=False, attention=True, attn_layer=2)


write_loss(window_length=90, cell_num=16, dense_dim=64,
         bidirection=False, attention=True, attn_layer=5)
# exit(0)

# train_model(window_length=90, epochs_num=epochs_num,
#      cell_num=16, dense_dim=64,
#      bidirection=False, attention=True, attn_layer=5, continue_epoch=-1)

attack_ids = [i for i in range(len(feature_names))]

model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
           threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = \
    detect_anomalies(window_length=90, cell_num=16, dense_dim=64, bidirection=False, attention=False,
                 attn_layer=5, attack_ids=attack_ids, avg_wl=3)
print(model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per, threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration)
model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
           threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = \
    detect_anomalies(window_length=90, cell_num=16, dense_dim=64, bidirection=False, attention=True,
                 attn_layer=5, attack_ids=attack_ids, avg_wl=3)
print(model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per, threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration)


# model_names = []
# threshold_TP_rates = []
# threshold_TP_duration_pers = []
# threshold_TP_avg_delays = []
# threshold_FP_pers = []
# train_model(window_length=best_window_length, epochs_num=20,
#      cell_num=best_cell_num, dense_dim=64,
#      bidirection=False, attention=False, attn_layer=1, continue_epoch=20)

# model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
#            threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = \
#     detect_anomalies(window_length=best_window_length, cell_num=best_cell_num, dense_dim=64, bidirection=False, attention=False,
#                  attn_layer=1, attack_ids=attack_ids, avg_wl=3)
# print(model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per, threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration)
#
# model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
#            threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = \
#     detect_anomalies(window_length=best_window_length, cell_num=best_cell_num, dense_dim=64, bidirection=False, attention=True,
#                  attn_layer=1, attack_ids=attack_ids, avg_wl=3)
# print(model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per, threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration)
#
# model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
#            threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = \
#     detect_anomalies(window_length=best_window_length, cell_num=best_cell_num, dense_dim=64, bidirection=False, attention=True,
#                  attn_layer=2, attack_ids=attack_ids, avg_wl=3)
# print(model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per, threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration)


# model_names.append(model_name)
# threshold_TP_rates.append(threshold_anomaly_TP_rate)
# threshold_TP_duration_pers.append(threshold_anomaly_detect_duration_per)
# threshold_TP_avg_delays.append(threshold_anomaly_detect_delay_avg)
# threshold_FP_pers.append(threshold_FP_per)
#
# model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
#            threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = \
#     detect_anomalies(window_length=best_window_length, cell_num=best_cell_num, dense_dim=64, bidirection=True, attention=False,
#                  attn_layer=1, attack_ids=[i for i in range(len(features))], avg_wl=3)
#
# model_names.append(model_name)
# threshold_TP_rates.append(threshold_anomaly_TP_rate)
# threshold_TP_duration_pers.append(threshold_anomaly_detect_duration_per)
# threshold_TP_avg_delays.append(threshold_anomaly_detect_delay_avg)
# threshold_FP_pers.append(threshold_FP_per)
#
# model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
#            threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = \
#     detect_anomalies(window_length=best_window_length, cell_num=best_cell_num, dense_dim=64, bidirection=False, attention=True,
#                  attn_layer=1, attack_ids=[i for i in range(len(features))], avg_wl=3)
#
# model_names.append(model_name)
# threshold_TP_rates.append(threshold_anomaly_TP_rate)
# threshold_TP_duration_pers.append(threshold_anomaly_detect_duration_per)
# threshold_TP_avg_delays.append(threshold_anomaly_detect_delay_avg)
# threshold_FP_pers.append(threshold_FP_per)
#
# model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
#            threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = \
#     detect_anomalies(window_length=best_window_length, cell_num=best_cell_num, dense_dim=64, bidirection=True, attention=True,
#                  attn_layer=1, attack_ids=[i for i in range(len(features))], avg_wl=3)
#
# model_names.append(model_name)
# threshold_TP_rates.append(threshold_anomaly_TP_rate)
# threshold_TP_duration_pers.append(threshold_anomaly_detect_duration_per)
# threshold_TP_avg_delays.append(threshold_anomaly_detect_delay_avg)
# threshold_FP_pers.append(threshold_FP_per)
#
# model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
#            threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = \
#     detect_anomalies(window_length=best_window_length, cell_num=best_cell_num, dense_dim=64, bidirection=True, attention=True,
#                  attn_layer=2, attack_ids=[i for i in range(len(features))], avg_wl=3)
#
# model_names.append(model_name)
# threshold_TP_rates.append(threshold_anomaly_TP_rate)
# threshold_TP_duration_pers.append(threshold_anomaly_detect_duration_per)
# threshold_TP_avg_delays.append(threshold_anomaly_detect_delay_avg)
# threshold_FP_pers.append(threshold_FP_per)
#
# model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
#            threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = \
#     detect_anomalies(window_length=best_window_length, cell_num=best_cell_num, dense_dim=64, bidirection=True, attention=True,
#                  attn_layer=4, attack_ids=[i for i in range(len(features))], avg_wl=3)
#
# model_names.append(model_name)
# threshold_TP_rates.append(threshold_anomaly_TP_rate)
# threshold_TP_duration_pers.append(threshold_anomaly_detect_duration_per)
# threshold_TP_avg_delays.append(threshold_anomaly_detect_delay_avg)
# threshold_FP_pers.append(threshold_FP_per)
#
# model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
#            threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = \
#     detect_anomalies(window_length=best_window_length, cell_num=best_cell_num, dense_dim=64, bidirection=True, attention=True,
#                  attn_layer=8, attack_ids=[i for i in range(len(features))], avg_wl=3)
#
# model_names.append(model_name)
# threshold_TP_rates.append(threshold_anomaly_TP_rate)
# threshold_TP_duration_pers.append(threshold_anomaly_detect_duration_per)
# threshold_TP_avg_delays.append(threshold_anomaly_detect_delay_avg)
# threshold_FP_pers.append(threshold_FP_per)
#
# df = pd.DataFrame(np.hstack([np.array(model_names).reshape([-1,1]), np.array(threshold_TP_rates), np.array(threshold_TP_duration_pers),
#                              np.array(threshold_TP_avg_delays), np.array(threshold_FP_pers)]),
#                        columns=['model_names']+[f'TPR_{i}' for i in range(5)]+[f'TP_percentage{i}' for i in range(5)]+[f'TP_avg_delay_{i}' for i in range(5)]+[f'FP_percentage{i}' for i in range(5)])
# file_path = os.path.join(ROOT_DIR, "results", feature_set, f'detection_res-{best_cell_num}cell-wl{best_window_length}.csv')
# df.to_csv(file_path, index=False)
#
# plot_detect_res(file_path)