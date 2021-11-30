import math
import os

# from classic_rnn.Model import plot_detect_res

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from keras_multi_head import MultiHead
import pandas as pd

tf.get_logger().setLevel('ERROR')

tf.keras.backend.set_floatx('float64')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from sklearn.preprocessing import MinMaxScaler

from tools.anomaly_creator import insert_super_anomalies, print_all, sample_from_np, detect_results_to_spans, \
    compare_with_threshold, span_analyze

# omit_features = ['Throttle_Control', 'Servo_Control', 'Voltage', 'Throttle_A', 'Throttle_B', 'Servo', 'Linear_Y', 'Reference']
# remain_features = ['Linear_X', 'Linear_Z', 'Acceleration_X', 'Acceleration_Y', 'Acceleration_Z', 'Angular_X', 'Angular_Y', 'Angular_Z']
# remain_feature_ids = [6, 8, 9, 10, 11, 12, 13, 14]
# id_project = dict(zip(remain_feature_ids, range(len(remain_feature_ids))))

# omit_features = ['throttl_m1', 'throttl_m2', 'throttl_m3', 'throttl_m4', 'throttle', 'throttle_out']
# remain_features = ['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'x_velocity', 'y_velocity', 'z_velocity',
#                    'yaw_angle', 'pitch_angle', 'roll_angle', 'yaw_angle_velocity', 'pitch_angle_velocity', 'roll_angle_velocity']
# remain_feature_ids = [i for i in range(6, 21)]
# id_project = dict(zip(remain_feature_ids, range(len(remain_feature_ids))))

omit_features = ['Reference', 'Control']
remain_features = ['Measure', 'Sate_1', 'Sate_2', 'Sate_3']
remain_feature_ids = [1,2,3,4]
id_project = dict(zip(remain_feature_ids, range(len(remain_feature_ids))))

def read_csv(file_path):
    df = pd.read_csv(file_path)
    try:
        df.drop(['Reference'], inplace=True, axis=1)
    except KeyError as e:
        pass

    feature_names = df.columns
    # print(f'feature_names: {feature_names}')
    # remain_feature_ids = []
    # for name in remain_features:
    #     remain_feature_ids.append(df.columns.get_loc(name))
    # remain_feature_names = remain_features
    # remain_feature_ids = [6,8,9,10,11,12,13,14]
    # return df, feature_names, remain_feature_ids
    return df, feature_names

def get_scalers_df(df):
    scalers = dict()
    for col in df.columns:
        col_df = df[col].to_numpy().reshape((-1, 1))
        scaler = MinMaxScaler()
        scaler.fit(col_df)
        scalers[col] = scaler
    return scalers

def scale_time_serials_df(df, scalers):
    normalized_time_serials = []
    for col in df.columns:
        col_df = df[col].to_numpy().reshape((-1, 1))
        scaler = scalers[col]
        normalized_time_serials.append(scaler.transform(col_df))
    normalized_time_serials = np.hstack(normalized_time_serials)
    return normalized_time_serials

def serials_to_samples(time_serials, remain_feature_ids=None, window_length=50, interval=1, jump=0):
    inputs = []
    targets = []
    size = time_serials.shape[0]
    for i in range(0, size - window_length - 1 - jump, interval):
        a_sample = time_serials[i:i + window_length]
        inputs.append(a_sample)
        targets.append([time_serials[i+window_length+jump, remain_feature_ids]])
    return np.array(inputs), np.array(targets)

def partition_df(df, split):
    size = df.shape[0]
    part_one = math.floor((split[0] / sum(split)) * size)

    if split[2] != 0:
        part_two = math.floor(((split[0] + split[1]) / sum(split)) * size)
        train_df = df.iloc[:part_one,:]
        test_df = df.iloc[part_one: part_two,:]
        attack_df = df.iloc[part_two:,:]
        # print(f'train_df: {train_df.shape}, test_df: {test_df.shape}, attack_df: {attack_df.shape}')
        return train_df, test_df, attack_df
    else:
        train_df = df.iloc[:part_one,:]
        test_df = df.iloc[part_one:,:]
        return train_df, test_df

def read_serials_from_csv(file_path, split, attack_ids=None, window_size=50):
    df, feature_names = read_csv(file_path)
    if split[2] != 0:
        train_df, test_df, attack_df = partition_df(df, split)
    else:
        train_df, test_df = partition_df(df, split)

    train_scalers = get_scalers_df(train_df)
    normalized_train_serials = scale_time_serials_df(train_df, train_scalers)
    normalized_test_serials = scale_time_serials_df(test_df, train_scalers)

    if split[2] != 0:
        anomaly_serials, anomalies = insert_super_anomalies(attack_df.to_numpy(), feature_ids=attack_ids, max_anom_duration=50, cooldown=50, window_len=window_size, avg_anomaly_interval=100)
        # anomaly_serials, anomalies = insert_super_anomalies(attack_df.to_numpy(), attack_ids, 50, 50, 200)
        # anomaly_serials, anomalies = insert_super_anomalies(attack_df.to_numpy(), remain_feature_ids, 50, 50)
        attack_df = pd.DataFrame(anomaly_serials, columns=feature_names)
        normalized_attack_serials = scale_time_serials_df(attack_df, train_scalers)
        for anomaly in anomalies:
            anomaly[2] = id_project[anomaly[2]]
        return normalized_train_serials, normalized_test_serials, normalized_attack_serials, anomalies, feature_names, remain_feature_ids
    else:
        return normalized_train_serials, normalized_test_serials, feature_names, remain_feature_ids

def sample_from_csv(file_path, window_size=50, interval=10, jump=0, split=(8,2,0), attack_ids=None):
    if split[2] != 0:
        normalized_train_serials, normalized_test_serials, normalized_attack_serials, anomalies, feature_names, remain_feature_ids = read_serials_from_csv(file_path, split, attack_ids, window_size)

        train_inputs, train_targets = serials_to_samples(normalized_train_serials,
                                                         remain_feature_ids=remain_feature_ids,
                                                         window_length=window_size, interval=interval, jump=jump)
        test_inputs, test_targets = serials_to_samples(normalized_test_serials, remain_feature_ids=remain_feature_ids,
                                                       window_length=window_size, interval=interval, jump=jump)
        attack_inputs, _ = serials_to_samples(normalized_attack_serials, remain_feature_ids=remain_feature_ids,
                                                       window_length=window_size, interval=interval, jump=jump)
        return train_inputs, train_targets, test_inputs, test_targets, attack_inputs, normalized_attack_serials[:, remain_feature_ids], anomalies, feature_names
    else:
        normalized_train_serials, normalized_test_serials, feature_names, remain_feature_ids = read_serials_from_csv(file_path, split)
        train_inputs, train_targets = serials_to_samples(normalized_train_serials,
                                                         remain_feature_ids=remain_feature_ids,
                                                         window_length=window_size, interval=interval, jump=jump)
        test_inputs, test_targets = serials_to_samples(normalized_train_serials, remain_feature_ids=remain_feature_ids,
                                                       window_length=window_size, interval=interval, jump=jump)
        return train_inputs, train_targets, test_inputs, test_targets, feature_names


def loss_fn(y_true, y_pred):
    # return tf.reduce_mean(tf.abs(tf.subtract(tf.reshape(y_true, [-1, y_true.shape[2]]), y_pred)), axis=1)
    return tf.keras.metrics.mean_squared_logarithmic_error(y_true, y_pred)
    # return tf.keras.metrics.mean_squared_logarithmic_error(y_true, y_pred)

def define_threshold(df):
    df = pd.DataFrame(np.sort(df.values, axis=0), index=df.index, columns=df.columns)
    size = df.shape[0]
    percents = np.arange(9, 11) * math.floor(0.1 * size)
    thresholds = df.iloc[percents].to_numpy()
    end = thresholds[-1]
    # thresholds = np.vstack((thresholds, np.array([end*1.1, end*1.2, end*1.3])))
    thresholds = np.array([end*0.8, end*0.9, end, end*1.1, end*1.2])
    return thresholds


def load_model_weights(window_length=50, jump=0, batch_size=64,
                       input_feature_num=15, target_feature_num=8, cell_num=128, dense_dim=64,
                       bidirection=False, attention=True, attn_layer=1, scenario='pid_kf'):
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
    model.add(layers.Dense(target_feature_num))
    model.build(input_shape=(batch_size, window_length, input_feature_num))
    model.compile(optimizer='adam', loss='mse')
    # model.compile(optimizer='adam', loss=loss_fn)

    model_name = f'{cell_num}cell-{"Bi" if bidirection else "Uni"}-LSTM-{f"{attn_layer}lyrs-Attn-" if attention else ""}wl{window_length}-jp{jump}-{dense_dim}'
    model_weight_path = os.path.join(ROOT_DIR, 'models_weights', 'testbed_models', scenario, f'{model_name}/checkpoint')
    if os.path.exists(model_weight_path):
        print("Load Model")
        model.load_weights(model_weight_path)
    return model, model_name, model_weight_path

def print_loss(title, losses, features, path):
    plt.figure(figsize=(30, 16))
    plt.suptitle(f'loss - {title}', fontsize=30)
    for i in range(len(features)):
        plt.subplot(math.ceil(len(features) / 2), 2, i + 1)
        feature_loss = losses[:, i]
        plt.hist(x=feature_loss, bins='auto')
        plt.grid(axis='y')
        plt.title(f'{features[i]}')
    for i in range(1, 10):
        file_name = f'loss - {title}.png'
        if not os.path.exists(os.path.join(path, file_name)):
            plt.savefig(os.path.join(path, file_name))
            # plt.show()
            break

def detect_anomalies(file_path, window_length=50, jump=0, batch_size=64,
                     cell_num=128, dense_dim=64, epochs=50,
                     bidirection=True, attention=True, attn_layer=4, scenario='pid_kf', split=(6,2,2), attack_ids=None, avg_wl=3, detect=True):

    # serials_df = pd.read_csv(os.path.join(ROOT_DIR, "results", 'sub_corelated_features', '20181203_Driver1_Trip10_Anomalous_Serials.csv'))
    # anomaly_serials = serials_df.to_numpy(dtype=float)
    # anomalies_df = pd.read_csv(os.path.join(ROOT_DIR, "results", 'sub_corelated_features', '20181203_Driver1_Trip10_Anomalies.csv'))
    # anomalies = anomalies_df.values.tolist()

    train_inputs, train_targets, test_inputs, test_targets, attack_inputs, normalized_attack_serials, anomalies, feature_names = \
        sample_from_csv(file_path, window_size=window_length, interval=1, jump=jump, split=split, attack_ids=attack_ids)
    train_targets = np.reshape(train_targets, [-1, train_targets.shape[2]])
    test_targets = np.reshape(test_targets, [-1, test_targets.shape[2]])

    print(
        f'train_inputs:{train_inputs.shape},train_targets:{train_targets.shape},test_inputs:{test_inputs.shape},test_targets:{test_targets.shape}')
    print(f'normalized_attack_serials shape: {normalized_attack_serials.shape}')
    print(f'anomalies:')
    print(anomalies)

    model, model_name, model_weight_path = load_model_weights(window_length=window_length, jump=jump,
                                                              batch_size=batch_size,
                                                              scenario=scenario, input_feature_num=len(feature_names),
                                                              target_feature_num=len(remain_features),
                                                              cell_num=cell_num,
                                                              dense_dim=dense_dim,
                                                              bidirection=bidirection, attention=attention,
                                                              attn_layer=attn_layer)
    print(f'model_name: {model_name}')
    if not detect:
        if epochs > 0:
            evaluation_res = []
            train_res = []
            validate_res = []
            for i in range(epochs):
                h = model.fit(train_inputs, train_targets, validation_split=0.1, batch_size=batch_size, epochs=1,
                              shuffle=True, verbose=0)
                eval_loss = model.evaluate(test_inputs, test_targets, verbose=0)
                train_loss = h.history['loss'][-1]
                vali_loss = h.history['val_loss'][-1]
                evaluation_res.append(eval_loss)
                train_res.append(train_loss)
                validate_res.append(vali_loss)
                print(f"{i} - train_loss:{train_loss},val_loss:{vali_loss},eval_loss:{eval_loss}")
            model.save_weights(model_weight_path)

            history_csv_path = os.path.join(ROOT_DIR, "results", 'testbed', scenario, f'history-{model_name}.csv')
            train_res = np.array(train_res).reshape([-1, 1])
            validate_res = np.array(validate_res).reshape([-1, 1])
            evaluation_res = np.array(evaluation_res).reshape([-1, 1])
            records = pd.DataFrame(np.hstack([train_res, validate_res, evaluation_res]),
                                   columns=['train_loss', 'val_loss', 'eval_loss'])

            if os.path.exists(history_csv_path):
                history = pd.read_csv(history_csv_path)
                records = pd.concat([history, records], ignore_index=True)
            print(f'loss records of {model_name}')
            records.to_csv(history_csv_path, index=False)

        if epochs >= 0:
            outputs = []
            for i in range(0, test_inputs.shape[0], 100):
                outputs.append(model(test_inputs[i:i + 100, :, :]))
            outputs = np.vstack(outputs)
            train_losses = np.abs(outputs - test_targets)
            df = pd.DataFrame(data=train_losses, columns=remain_features)
            df.to_csv(os.path.join(ROOT_DIR, "results", 'testbed', scenario, f'loss-{model_name}.csv'), index=False)

            res = model.evaluate(test_inputs, test_targets)
            print(res)

            print_loss(model_name, train_losses, remain_features, os.path.join(ROOT_DIR, "results", 'testbed', scenario))

        # return

    threshold_csv_path = os.path.join(ROOT_DIR, 'results', 'testbed', scenario, f'loss-{model_name}.csv')
    df = pd.read_csv(threshold_csv_path)
    thresholds = define_threshold(df)

    outputs = []
    for i in range(0, attack_inputs.shape[0], 100):
        outputs.append(model(attack_inputs[i:i + 100, :, :]))
    outputs = np.vstack(outputs)
    outputs = np.vstack((normalized_attack_serials[:window_length+jump+1], outputs))
    print(f'outputs shape: {outputs.shape}')

    losses = np.abs(normalized_attack_serials - outputs)
    print(f'losses shape: {outputs.shape}')

    losses_copy = np.copy(losses)
    # print(f'losses_copy shape: {losses_copy.shape}')
    avg_losses = []
    for i in range(losses_copy.shape[1]):
        a_loss = losses_copy[:, i].reshape(-1)
        # print(a_loss.shape)
        a_avg_loss = np.convolve(a_loss, np.ones(avg_wl) / avg_wl, mode='valid')
        # print(a_avg_loss.shape)
        avg_losses.append(a_avg_loss.reshape([-1, 1]))

    avg_losses = np.hstack(avg_losses)
    avg_losses = np.vstack((np.zeros([avg_wl-1, avg_losses.shape[1]]).astype(bool), avg_losses))
    print(avg_losses.shape)


    results = compare_with_threshold(avg_losses, thresholds)
    # results = compare_with_threshold_max(avg_losses, thresholds)

    print(f'results shape: {results.shape}')
    # print(np.all(results==False))
    spans = detect_results_to_spans(results)  # shape: [feature_size, threshold_size, ...]
    print(spans)

    threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per, threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = span_analyze(spans, anomalies, normalized_attack_serials.shape[0])


    path = os.path.join(ROOT_DIR, "results", 'testbed', scenario, f'attack-{model_name}')
    os.makedirs(path, exist_ok=True)
    print_all(normal_serials=normalized_attack_serials, anomaly_serials=normalized_attack_serials, anomalies=anomalies,
              outputs=outputs, feature_names=remain_features, spans=spans,
              path=path)

    return f'{"Bi" if bidirection else "Uni"} {f"{attn_layer}lyr" if attention else "No"}-Attn', threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
           threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration

def lstm(window_len=50, sample_interval=10,
         jump=0, batch_size=64, epochs_num=20,
         attention=False, attn_layer=12,
         dense_dim=32, cell_num=128, bi=True,
         test_only=False, data_file=None, scenario='bare', plot_loss=False):
    train_inputs, train_targets, train_normalized_time_serials, features, train_scalers = sample_from_csv(
        data_file, window_size=window_len, interval=sample_interval, jump=jump)
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

    model_name = f'{cell_num}cell-{"Bi" if bi else "Uni"}-LSTM-{f"{attn_layer}lyrs-Attn-" if attention else ""}wl{window_len}-jp{jump}-{dense_dim}'
    model_weight_path = os.path.join(ROOT_DIR, "models_weights", "testbed_models", scenario, f'{model_name}/checkpoint')
    complete_model_path = os.path.join(ROOT_DIR, "complete_models", "testbed_models", scenario, model_name)

    if os.path.exists(model_weight_path):
        print("Load Model")
        model.load_weights(model_weight_path)

    if not test_only:
        model.fit(train_inputs, train_targets, validation_split=0.2, batch_size=batch_size, epochs=epochs_num,
                  shuffle=True)
        # model.fit(inputs_train, targets_train, validation_split=0.2, batch_size=batch_size, epochs=epochs_num, shuffle=True, sample_weight=sample_weights)
        model.save_weights(model_weight_path)
        model.save(complete_model_path)

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

# scenario = 'pid_kf'
# data_file = 'canvas_semi_auto_pid_kf.csv'
# file_path = os.path.join(ROOT_DIR, '..', 'data', 'testbed', 'barnes', data_file)
# detect_anomalies(file_path=file_path, window_length=50, jump=0, batch_size=64,
#                      cell_num=128, dense_dim=64, epochs=100,
#                      bidirection=False, attention=False, attn_layer=4, scenario=scenario, split=(6,2,2), attack_ids=remain_feature_ids) # [6,8,14]
#
# scenario = 'pid_kf'
# data_file = 'canvas_semi_auto_pid_kf.csv'
# file_path = os.path.join(ROOT_DIR, '..', 'data', 'testbed', 'barnes', data_file)
# detect_anomalies(file_path=file_path, window_length=50, jump=0, batch_size=64,
#                      cell_num=128, dense_dim=64, epochs=100,
#                      bidirection=True, attention=False, attn_layer=4, scenario=scenario, split=(6,2,2), attack_ids=remain_feature_ids) # [6,8,14]
#

epoch_num = 30
batch_size = 128
wl = 60
cell_num = 32
# scenario = 'pid_kf'
# data_file = 'canvas_semi_auto_pid_kf.csv'
# file_path = os.path.join(ROOT_DIR, '..', 'data', 'testbed', 'barnes', data_file)

scenario = 'simu'
data_file = 'aircraft_pitch.csv'
file_path = os.path.join(ROOT_DIR, '..', 'data', 'testbed', 'simulator', data_file)
attack_ids = remain_feature_ids

# detect_anomalies(file_path=file_path, window_length=wl, batch_size=batch_size,
#                      cell_num=cell_num, dense_dim=64, epochs=epoch_num,
#                      bidirection=False, attention=False, attn_layer=1, scenario=scenario, split=(7,2,1), attack_ids=attack_ids, detect=False) # [6,8,14]
# detect_anomalies(file_path=file_path, window_length=wl, batch_size=batch_size,
#                      cell_num=cell_num, dense_dim=64, epochs=epoch_num,
#                      bidirection=True, attention=False, attn_layer=1, scenario=scenario, split=(7,2,1), attack_ids=attack_ids, detect=False) # [6,8,14]
# detect_anomalies(file_path=file_path, window_length=wl, batch_size=batch_size,
#                      cell_num=cell_num, dense_dim=64, epochs=epoch_num,
#                      bidirection=False, attention=True, attn_layer=1, scenario=scenario, split=(7,2,1), attack_ids=attack_ids, detect=False) # [6,8,14]
# detect_anomalies(file_path=file_path, window_length=wl, batch_size=batch_size,
#                      cell_num=cell_num, dense_dim=64, epochs=epoch_num,
#                      bidirection=True, attention=True, attn_layer=1, scenario=scenario, split=(7,2,1), attack_ids=attack_ids, detect=False) # [6,8,14]
# detect_anomalies(file_path=file_path, window_length=wl, batch_size=batch_size,
#                      cell_num=cell_num, dense_dim=64, epochs=epoch_num,
#                      bidirection=True, attention=True, attn_layer=2, scenario=scenario, split=(7,2,1), attack_ids=attack_ids, detect=False) # [6,8,14]
# detect_anomalies(file_path=file_path, window_length=wl, batch_size=batch_size,
#                      cell_num=cell_num, dense_dim=64, epochs=epoch_num,
#                      bidirection=True, attention=True, attn_layer=4, scenario=scenario, split=(7,2,1), attack_ids=attack_ids, detect=False) # [6,8,14]
# detect_anomalies(file_path=file_path, window_length=wl, batch_size=batch_size,
#                      cell_num=cell_num, dense_dim=64, epochs=epoch_num,
#                      bidirection=True, attention=True, attn_layer=8, scenario=scenario, split=(7,2,1), attack_ids=attack_ids, detect=False) # [6,8,14]
# detect_anomalies(file_path=file_path, window_length=wl, batch_size=batch_size,
#                      cell_num=cell_num, dense_dim=64, epochs=epoch_num,
#                      bidirection=True, attention=True, attn_layer=16, scenario=scenario, split=(7,2,1), attack_ids=attack_ids, detect=False) # [6,8,14]
#
# exit(0)

# model_names = []
# threshold_TP_rates = []
# threshold_TP_avg_delays = []
# threshold_FP_pers = []
model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
           threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = detect_anomalies(file_path=file_path, window_length=wl, batch_size=batch_size,
                     cell_num=cell_num, dense_dim=64, epochs=epoch_num,
                     bidirection=False, attention=False, attn_layer=1, scenario=scenario, split=(7,2,1), attack_ids=attack_ids, detect=True) # [6,8,14]
print(model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per, threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration)

model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
           threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = detect_anomalies(file_path=file_path, window_length=wl, batch_size=batch_size,
                     cell_num=cell_num, dense_dim=64, epochs=epoch_num,
                     bidirection=False, attention=True, attn_layer=1, scenario=scenario, split=(7,2,1), attack_ids=attack_ids, detect=True) # [6,8,14]
print(model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per, threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration)

model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per,\
           threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = detect_anomalies(file_path=file_path, window_length=wl, batch_size=batch_size,
                     cell_num=cell_num, dense_dim=64, epochs=epoch_num,
                     bidirection=False, attention=True, attn_layer=4, scenario=scenario, split=(7,2,1), attack_ids=attack_ids, detect=True) # [6,8,14]
print(model_name, threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per, threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration)


# df = pd.DataFrame(np.hstack([np.array(model_names).reshape([-1,1]), np.array(threshold_TP_rates), np.array(threshold_TP_avg_delays), np.array(threshold_FP_pers)]),
#                        columns=['model_names']+[f'TPR_{i}' for i in range(5)]+[f'TP_avg_delay_{i}' for i in range(5)]+[f'FP_percentage{i}' for i in range(5)])
# file_path = os.path.join(ROOT_DIR, "results", 'testbed', scenario, f'detection_res-{cell_num}cell-wl{wl}.csv')
#
# df.to_csv(file_path, index=False)
#
# plot_detect_res(file_path)


# scenario = 'pid_kf'
# data_file = 'canvas_semi_auto_pid_kf.csv'
# file_path = os.path.join(ROOT_DIR, '..', 'data', 'testbed', 'barnes', data_file)
# detect_anomalies(file_path=file_path, window_length=50, jump=0, batch_size=64,
#                      cell_num=128, dense_dim=64, epochs=100,
#                      bidirection=True, attention=True, attn_layer=2, scenario=scenario, split=(6,2,2), attack_ids=remain_feature_ids) # [6,8,14]
#
# scenario = 'pid_kf'
# data_file = 'canvas_semi_auto_pid_kf.csv'
# file_path = os.path.join(ROOT_DIR, '..', 'data', 'testbed', 'barnes', data_file)
# detect_anomalies(file_path=file_path, window_length=50, jump=0, batch_size=64,
#                      cell_num=128, dense_dim=64, epochs=10,
#                      bidirection=True, attention=True, attn_layer=4, scenario=scenario, split=(6,2,2), attack_ids=remain_feature_ids) # [6,8,14]
#
# scenario = 'pid_kf'
# data_file = 'canvas_semi_auto_pid_kf.csv'
# file_path = os.path.join(ROOT_DIR, '..', 'data', 'testbed', 'barnes', data_file)
# detect_anomalies(file_path=file_path, window_length=50, jump=0, batch_size=64,
#                      cell_num=128, dense_dim=64, epochs=50,
#                      bidirection=True, attention=True, attn_layer=10, scenario=scenario, split=(6,2,2), attack_ids=remain_feature_ids) # [6,8,14]

# scenario = 'pid_kf'
# data_file = 'canvas_semi_auto_pid_kf.csv'
# file_path = os.path.join(ROOT_DIR, '..', 'data', 'testbed', 'barnes', data_file)
# detect_anomalies(file_path=file_path, window_length=50, jump=0, batch_size=64,
#                      cell_num=128, dense_dim=64, epochs=50,
#                      bidirection=True, attention=True, attn_layer=20, scenario=scenario, split=(6,2,2), attack_ids=remain_feature_ids) # [6,8,14]

# scenario = 'ardu'
# data_file = 'U4.csv'
# file_path = os.path.join(ROOT_DIR, '..', 'data', 'testbed', 'ardupilot', data_file)
# detect_anomalies(file_path=file_path, window_length=50, jump=0, batch_size=64,
#                      cell_num=128, dense_dim=64, epochs=-1,
#                      bidirection=True, attention=True, attn_layer=4, scenario=scenario, split=(8,1,1), attack_ids=remain_feature_ids)
#
# scenario = 'simu'
# data_file = 'aircraft_pitch.csv'
# file_path = os.path.join(ROOT_DIR, '..', 'data', 'testbed', 'simulator', data_file)
# detect_anomalies(file_path=file_path, window_length=50, jump=0, batch_size=64,
#                      cell_num=128, dense_dim=64, epochs=40,
#                      bidirection=True, attention=True, attn_layer=4, scenario=scenario, split=(7,2,1), attack_ids=remain_feature_ids)

# lstm(window_len=50, sample_interval=1,
#      jump=0, batch_size=64, epochs_num=50,
#      bi=False, attention=False, attn_layer=0,
#      cell_num=128, dense_dim=64,
#      test_only=True, data_file=data_file, scenario=scenario, plot_loss=True)
# lstm(window_len=50, sample_interval=1,
#      jump=0, batch_size=64, epochs_num=50,
#      bi=True, attention=False, attn_layer=0,
#      cell_num=128, dense_dim=64,
#      test_only=False, data_file=data_file, scenario=scenario, plot_loss=True)
# lstm(window_len=50, sample_interval=1,
#      jump=0, batch_size=64, epochs_num=50,
#      bi=False, attention=True, attn_layer=1,
#      cell_num=128, dense_dim=64,
#      test_only=False, data_file=data_file, scenario=scenario, plot_loss=True)
# lstm(window_len=50, sample_interval=1,
#      jump=0, batch_size=64, epochs_num=50,
#      bi=True, attention=True, attn_layer=1,
#      cell_num=128, dense_dim=64,
#      test_only=False, data_file=data_file, scenario=scenario, plot_loss=True)
# lstm(window_len=50, sample_interval=1,
#      jump=0, batch_size=64, epochs_num=50,
#      bi=True, attention=True, attn_layer=2,
#      cell_num=128, dense_dim=64,
#      test_only=False, data_file=data_file, scenario=scenario, plot_loss=True)
# lstm(window_len=50, sample_interval=1,
#      jump=0, batch_size=64, epochs_num=80,
#      bi=True, attention=True, attn_layer=4,
#      cell_num=128, dense_dim=64,
#      test_only=False, data_file=data_file, scenario=scenario, plot_loss=True)
