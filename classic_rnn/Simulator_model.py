import math
import os
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

simulator_model = 'aircraft_pitch'
# omit_features = ['Reference', 'Control']
# output_features = ['Sate_1', 'Sate_2', 'Sate_3']
# input_features = ['Control', 'Sate_1', 'Sate_2', 'Sate_3']

data_dir_path = os.path.join(ROOT_DIR, '..', 'data', 'simulator', simulator_model)
result_dir_path = os.path.join(ROOT_DIR, 'results', 'simulator', simulator_model)
model_weights_dir_path = os.path.join(ROOT_DIR, 'models_weights', 'simulator_models', simulator_model)

# id_project = dict(zip(remain_feature_ids, range(len(remain_feature_ids))))

jump = 0
batch_size = 64

def read_csv(file_path):
    df = pd.read_csv(file_path)
    feature_names = df.columns
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

def serials_to_samples(time_serials, window_length=50, interval=1, jump=0):
    inputs = []
    targets = []
    size = time_serials.shape[0]
    for i in range(0, size - window_length - 1 - jump, interval):
        inputs.append(time_serials[i:i + window_length, input_features])
        targets.append(time_serials[i+window_length+jump, output_features])
    return np.array(inputs), np.vstack(targets)

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

    df = pd.DataFrame(data=losses, columns=output_features)
    df.to_csv(os.path.join(result_dir_path, f'loss-{model_name}.csv'), index=False)

def train_model(window_length=50, epochs_num=10, cell_num=128, dense_dim=64,
         bidirection=False, attention=False, attn_layer=4):
    train_df, _ = read_csv(os.path.join(data_dir_path, 'aircraft_pitch.csv'))
    evaluation_df, _ = read_csv(os.path.join(data_dir_path, 'aircraft_pitch_vali.csv'))
    # test_df, _ = read_csv(os.path.join(data_dir_path, 'aircraft_pitch_test.csv'))

    scalers = get_scalers_df(train_df)

    normalized_train_serials = scale_time_serials_df(train_df, scalers)
    normalized_evaluation_serials = scale_time_serials_df(evaluation_df, scalers)
    # normalized_test_serials = scale_time_serials_df(test_df, scalers)

    train_inputs, train_targets = serials_to_samples(time_serials=normalized_train_serials, window_length=window_length)
    evaluation_inputs, evaluation_targets = serials_to_samples(time_serials=normalized_evaluation_serials, window_length=window_length)
    # test_inputs, test_targets = serials_to_samples(time_serials=normalized_test_serials, window_length=window_length)

    model, model_name, model_weight_path, epoch_num = load_model_weights(window_length=window_length, jump=jump, batch_size=batch_size,
                       input_feature_num=len(input_features), target_feature_num=len(output_features), cell_num=cell_num, dense_dim=dense_dim,
                       bidirection=bidirection, attention=attention, attn_layer=attn_layer)

    if epochs_num > 0:
        evaluation_res = []
        train_res = []
        epoch_nums = []
        for i in range(epochs_num):
            epoch_num = epoch_num+1+i
            h = model.fit(train_inputs, train_targets, batch_size=batch_size, epochs=1,
                          shuffle=True, verbose=0)
            eval_loss = model.evaluate(evaluation_inputs, evaluation_targets, verbose=0)
            train_loss = h.history['loss'][-1]
            evaluation_res.append(eval_loss)
            train_res.append(train_loss)
            epoch_nums.append(epoch_num)
            print(f"{i} - train_loss:{train_loss},eval_loss:{eval_loss}")
            model.save_weights(os.path.join(model_weights_dir_path, f'{model_name}', f'epoch{epoch_num}/checkpoint'))

        train_res = np.array(train_res).reshape([-1, 1])
        evaluation_res = np.array(evaluation_res).reshape([-1, 1])
        records = pd.DataFrame(np.hstack([epoch_nums, train_res, evaluation_res]),
                               columns=['epoch', 'train_loss', 'eval_loss'])

        history_csv_path = os.path.join(result_dir_path, f'history-{model_name}')
        records.to_csv(history_csv_path, index=False)

        best_epoch = find_lowest_loss_epoch(history_csv_path)
        print(f'best_epoch: {best_epoch}')

        model, model_name, model_weight_path, epoch_num = load_model_weights(window_length=window_length, jump=jump,
                                                                             batch_size=batch_size,
                                                                             input_feature_num=len(input_features),
                                                                             target_feature_num=len(output_features),
                                                                             cell_num=cell_num, dense_dim=dense_dim,
                                                                             bidirection=bidirection,
                                                                             attention=attention, attn_layer=attn_layer)

        print_train_loss(evaluation_inputs, evaluation_targets, model, model_name)

def define_threshold(model_name):
    losses = pd.read_csv(os.path.join(result_dir_path, f'loss-{model_name}.csv'), index = False)
    sorted_losses = np.sort(losses.values, axis=0)
    end = sorted_losses[-1]
    thresholds = np.array([end*0.8, end*0.9, end, end*1.1, end*1.2])
    return thresholds

def find_lowest_loss_epoch(model_name):
    try:
        history_df = pd.read_csv(os.path.join(result_dir_path, f'history-{model_name}'))
        row_id_min_eval_loss = history_df['eval_loss'].idxmin()
        return row_id_min_eval_loss+1
    except:
        return 0

def load_model_weights(window_length=50, jump=0, batch_size=64,
                       input_feature_num=15, target_feature_num=8, cell_num=128, dense_dim=64,
                       bidirection=False, attention=True, attn_layer=1):
    model = keras.Sequential()
    a_layer = layers.LSTM(cell_num, return_sequences=False)
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

    best_epoch = find_lowest_loss_epoch(model_name)

    model_weight_path = os.path.join(model_weights_dir_path, f'{model_name}', f'epoch{best_epoch}/checkpoint')
    if os.path.exists(model_weight_path):
        print("Load Model")
        model.load_weights(model_weight_path)
    return model, model_name, model_weight_path, best_epoch

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
                     bidirection=True, attention=True, attn_layer=4, attack_ids=None, avg_wl=3):
    model, model_name, model_weight_path, epoch_num = load_model_weights(window_length=window_length, jump=jump,
                                                                         batch_size=batch_size,
                                                                         input_feature_num=len(input_features),
                                                                         target_feature_num=len(output_features),
                                                                         cell_num=cell_num, dense_dim=dense_dim,
                                                                         bidirection=bidirection,
                                                                         attention=attention, attn_layer=attn_layer)

    train_df, _ = read_csv(os.path.join(data_dir_path, 'aircraft_pitch.csv'))
    anomaly_df, _ = read_csv(os.path.join(data_dir_path, 'aircraft_pitch_Anomalous_Serials.csv'))
    scalers = get_scalers_df(train_df)

    normalized_anomaly_serials = scale_time_serials_df(anomaly_df, scalers)
    anomaly_inputs, anomaly_targets = serials_to_samples(time_serials=normalized_anomaly_serials, window_length=window_length)

    print(f'normalized_attack_serials shape: {normalized_attack_serials.shape}')
    print(f'anomalies:')
    print(anomalies)

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
