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
do_log = True

simulator_model = 'aircraft_pitch'
all_features = ['ref', 'x1', 'x2', 'x3', 'cin']
input_features = ['x1', 'x2', 'x3', 'cin']
input_feature_ids = [1, 2, 3, 4]
output_features = ['x1', 'x2', 'x3']
output_feature_ids = [1, 2, 3]
real_features = ['ux1', 'ux2', 'ux3']
real_feature_ids = [5, 6, 7]
cin_features = ['cin']
cin_features_ids = [4]

data_dir_path = os.path.join(ROOT_DIR, '..', 'data', 'simulator', simulator_model)
result_dir_path = os.path.join(ROOT_DIR, 'results', 'simulator', simulator_model)
model_weights_dir_path = os.path.join(ROOT_DIR, 'models_weights', 'simulator_models', simulator_model)

# id_project = dict(zip(remain_feature_ids, range(len(remain_feature_ids))))

jump = 0
batch_size = 64
sample_interval = 2

normalized_benign_file_names = [f for f in os.listdir(data_dir_path) if f[-3:] == "csv" and f[:13] == "scaled-benign"]
normalized_anomalous_file_names = [f for f in os.listdir(data_dir_path) if f[-3:] == "csv" and f[:16] == "scaled-anomalous"]

def log(content):
    if do_log:
        print(content)

def insert_noise(df, col_name, scale=0.01):
    length = df.shape[0]
    noise = np.random.normal(0, scale, length)
    df[col_name] += noise

def sample_from_df(dfs, window_length=50, interval=sample_interval, jump=jump):
    inputs = []
    targets = []
    for df in dfs:
        serials = df.to_numpy()
        length = serials.shape[0]
        for i in range(0, length - window_length - jump - 1, interval):
            inputs.append(serials[i:i+window_length, input_feature_ids])
            targets.append([serials[i+window_length+jump, output_feature_ids]])
    inputs = np.array(inputs)
    targets = np.vstack(targets)
    return inputs, targets

def load_model_weights(window_length=50, jump=jump, batch_size=batch_size,
                       input_feature_num=len(input_feature_ids), target_feature_num=len(output_feature_ids), cell_num=128, dense_dim=64,
                       bidirection=False, attn_layer=1, from_last = False):
    model = keras.Sequential()
    a_layer = layers.LSTM(cell_num, return_sequences=False)
    if bidirection:
        a_layer = layers.Bidirectional(a_layer)
    if attn_layer > 0:
        a_layer = MultiHead(a_layer, layer_num=attn_layer)
    model.add(a_layer)
    if attn_layer > 0:
        model.add(layers.Flatten())
    model.add(layers.Dense(dense_dim))
    model.add(layers.Dense(target_feature_num))
    model.build(input_shape=(batch_size, window_length, input_feature_num))
    model.compile(optimizer='adam', loss='mse')
    # model.compile(optimizer='adam', loss=loss_fn)

    model_name = f'{cell_num}cell-{"Bi" if bidirection else "Uni"}-LSTM-{f"{attn_layer}lyrs-Attn-" if attn_layer > 0 else ""}wl{window_length}-jp{jump}-{dense_dim}'

    if from_last:
        epoch, df = find_last_epoch(model_name)
    else:
        epoch, df = find_lowest_loss_epoch(model_name)

    model_weight_path = os.path.join(model_weights_dir_path, model_name, f'epoch{epoch}/checkpoint')
    if os.path.exists(model_weight_path):
        print(f"Load {model_name} from epoch {epoch}")
        model.load_weights(model_weight_path)

    return model, model_name, epoch, df

def find_lowest_loss_epoch(model_name):
    try:
        history_df = pd.read_csv(os.path.join(model_weights_dir_path, model_name, f'history.csv'))
        row_id_min_eval_loss = history_df['eval_loss'].idxmin()
        return row_id_min_eval_loss + 1, history_df
    except:
        return 0, pd.DataFrame(columns=['epoch', 'train_loss', 'eval_loss'])

def find_last_epoch(model_name):
    try:
        history_df = pd.read_csv(os.path.join(model_weights_dir_path, model_name, f'history.csv'))
        return history_df.shape[0], history_df
    except:
        return 0, pd.DataFrame(columns=['epoch', 'train_loss', 'eval_loss'])

def compute_outputs(inputs, model):
    outputs = []

    for i in range(0, inputs.shape[0], 100):
        outputs.append(model(inputs[i:i + 100]))
    outputs = np.vstack(outputs)

    return outputs

def record_vali_loss(inputs, targets, model, path):
    outputs = compute_outputs(inputs, model)

    assert outputs.shape == targets.shape

    # losses = np.abs(outputs - targets)
    losses = np.square(outputs - targets)

    df = pd.DataFrame(data=losses, columns=output_features)
    df.to_csv(os.path.join(path, 'loss.csv'), index=False)

def train_model(window_length=50, epochs=10, cell_num=128, dense_dim=64,
                bidirection=False, attn_layer=4):
    # attention = True if attn_layer > 0 else False
    validate_file_name = normalized_benign_file_names[-1]
    normalized_train_dfs = [pd.read_csv(os.path.join(data_dir_path, f)) for f in normalized_benign_file_names[:-1]]
    normalized_validate_df = pd.read_csv(os.path.join(data_dir_path, validate_file_name))
    log(f'Collected {len(normalized_train_dfs)} files for training; use {validate_file_name} as validation')

    train_inputs, train_targets = sample_from_df(normalized_train_dfs, window_length=window_length)
    validate_inputs, validate_targets = sample_from_df([normalized_validate_df], window_length=window_length)

    model, model_name, epoch_num_load, history_df = load_model_weights(window_length=window_length, cell_num=cell_num, dense_dim=dense_dim,
                                                                         bidirection=bidirection, attn_layer=attn_layer, from_last=True)
    if epochs > 0:
        evaluation_res = []
        train_res = []
        epoch_nums = []
        for i in range(epochs):
            epoch_num = epoch_num_load + 1 + i
            h = model.fit(train_inputs, train_targets, batch_size=batch_size, epochs=1,
                          shuffle=True, verbose=0)
            model.save_weights(os.path.join(model_weights_dir_path, model_name, f'epoch{epoch_num}/checkpoint'))

            eval_loss = model.evaluate(validate_inputs, validate_targets, verbose=0)
            record_vali_loss(validate_inputs, validate_targets, model, os.path.join(model_weights_dir_path, model_name, f'epoch{epoch_num}'))
            train_loss = h.history['loss'][-1]
            evaluation_res.append(eval_loss)
            train_res.append(train_loss)
            epoch_nums.append(epoch_num)
            log(f"{i} - train_loss:{train_loss}, eval_loss:{eval_loss}")

        epoch_nums = np.array(epoch_nums, dtype=int).reshape([-1, 1])
        train_res = np.array(train_res).reshape([-1, 1])
        evaluation_res = np.array(evaluation_res).reshape([-1, 1])
        new_history = pd.DataFrame(np.hstack([epoch_nums, train_res, evaluation_res]),
                               columns=['epoch', 'train_loss', 'eval_loss'])

        history_df = pd.concat([history_df, new_history], ignore_index=True)
        history_df.reset_index(drop=True, inplace=True)

        history_df.to_csv(os.path.join(model_weights_dir_path, model_name, f'history.csv'), index=False)

def define_threshold(path):
    losses = pd.read_csv(os.path.join(path, 'loss.csv'), index=False)
    # losses = pd.read_csv(os.path.join(result_dir_path, f'loss-{model_name}.csv'), index=False)
    sorted_losses = np.sort(losses.values, axis=0)
    end = sorted_losses[-1]
    thresholds = np.array([end * 0.8, end * 0.9, end, end * 1.1, end * 1.2])
    return thresholds

# Not Using
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

def plot_lines(**kwargs):
    real = kwargs.get("real", None)
    measurement = kwargs.get("measurement", None)  # real + anomaly
    prediction = kwargs.get("prediction", None)
    cin = kwargs.get("cin", None)
    ref = kwargs.get("ref", None)

    if real is None:
        print('Real is None, cannot plot')
        return

    if prediction is not None:
        assert real.shape == prediction.shape

    if measurement is not None:
        assert real.shape == measurement.shape

    if cin is not None:
        assert real.shape[0] == cin.shape[0]

    if ref is not None:
        assert real.shape[0] == ref.shape[0]

    plot_length = kwargs.get("plot_length", real.shape[0])
    feature_size = real.shape[1]

    x_axis = kwargs.get("x", np.arange(plot_length).reshape([-1,1]))
    one_graph_len = kwargs.get("one_graph_len", 400)
    feature_names = kwargs.get("feature_names", [f'feature-{i+1}' for i in range(feature_size)])

    if feature_size <= 3:
        rows = feature_size
        cols = 1
    else:
        cols = 2
        rows = math.ceil(feature_size / 2)

    ploted = 0
    while ploted < plot_length:
        to_plot = min(ploted + one_graph_len, plot_length)
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(28, 15))

        for i in range(feature_size):
            if cols == 1:
                ax[i].plot(x_axis[ploted:to_plot], measurement[ploted:to_plot, i], label='Measurement', c='r', marker='.')
                ax[i].plot(x_axis[ploted:to_plot], real[ploted:to_plot, i], label='Real', c='b', marker='.')
                ax[i].plot(x_axis[ploted:to_plot], prediction[ploted:to_plot, i], label='Prediction', c='g', marker='.')
                if i == feature_size-1:
                    ax[i].plot(x_axis[ploted:to_plot], cin[ploted:to_plot], label='cin', c='y', marker='.')
                    # ax[i].plot(x_axis, ref[ploted:to_plot], label='ref', c='orange', marker='.')
                ax[i].legend(loc="upper right")
                ax[i].set_title(f'{feature_names[i]}')
            else:
                row = math.floor(i / cols)
                col = i % cols
                ax[row, col].plot(x_axis[ploted:to_plot], measurement[ploted:to_plot, i], label='Measurement', c='r', marker='.')
                ax[row, col].plot(x_axis[ploted:to_plot], real[ploted:to_plot, i], label='Real', c='b', marker='.')
                ax[row, col].plot(x_axis[ploted:to_plot], prediction[ploted:to_plot, i], label='Prediction', c='g', marker='.')
                if i == feature_size-1:
                    ax[row, col].plot(x_axis[ploted:to_plot], cin[ploted:to_plot], label='cin', c='y', marker='.')
                ax[row, col].legend(loc="upper right")
                ax[row, col].set_title(f'{feature_names[i]}')

        show = kwargs.get("show", False)
        save_path = kwargs.get("save_path", None)
        if show:
            plt.show()
        if save_path is not None:
            name = kwargs.get("name", 'NONAME')
            plt.savefig(os.path.join(save_path, f'{name}-{ploted}-{to_plot}.png'))
        plt.close()
        ploted = to_plot

def concat_inputs(samples):
    serials = []
    for i in range(1, samples.shape[0]):
        serials.append(samples[i,-1,:])
    return np.vstack(serials)

def detect_anomalies(window_length=50, cell_num=128, dense_dim=64, bidirection=False, attn_layer=4):
    model, model_name, epoch_num_load, _ = load_model_weights(window_length=window_length, cell_num=cell_num, dense_dim=dense_dim,
                                                                     bidirection=bidirection, attn_layer=attn_layer, from_last=False)  # from the best
    log(f'Use epoch {epoch_num_load} of {model_name} for anomaly detection')
    for f in normalized_anomalous_file_names:
        log(f'Scenario {f}')
        complete_df = pd.read_csv(os.path.join(data_dir_path, f))
        model_df = complete_df[all_features]
        real_state_df = complete_df[real_features]
        cin_df = complete_df[cin_features]
        ref_df = complete_df['ref']
        inputs, targets = sample_from_df([model_df], window_length=window_length, interval=1)

        prediction = compute_outputs(inputs, model)
        real = real_state_df.to_numpy()
        cin = cin_df.to_numpy()
        ref = ref_df.to_numpy()

        # real = real[window_length:-1]
        # print(targets[:5])
        # print(real[:5])
        #
        # print('===')
        # print(targets[-5:])
        # print(real[-5:])

        # print(targets.shape)
        # print(real.shape)
        # print(real[window_length:].shape)
        # exit(0)

        plot_lines(measurement=targets, real=real[window_length:-1],
                   prediction=prediction, cin=cin[window_length:-1],
                   ref=ref[window_length+1:],
                   plot_length=400, show=False, save_path=result_dir_path, name=f'{model_name}-{f[:-4]}')
        # exit(0)



# def detect_anomalies(file_path, window_length=50, jump=0, batch_size=64,
#                      cell_num=128, dense_dim=64, epochs=50,
#                      bidirection=True, attention=True, attn_layer=4, attack_ids=None, avg_wl=3):
#     model, model_name, model_weight_path, epoch_num = load_model_weights(window_length=window_length, jump=jump,
#                                                                          batch_size=batch_size,
#                                                                          input_feature_num=len(input_features),
#                                                                          target_feature_num=len(output_features),
#                                                                          cell_num=cell_num, dense_dim=dense_dim,
#                                                                          bidirection=bidirection,
#                                                                          attention=attention, attn_layer=attn_layer)
#
#     train_df, _ = read_csv(os.path.join(data_dir_path, 'aircraft_pitch.csv'))
#     anomaly_df, _ = read_csv(os.path.join(data_dir_path, 'aircraft_pitch_Anomalous_Serials.csv'))
#     scalers = get_scalers_df(train_df)
#
#     normalized_anomaly_serials = scale_time_serials_df(anomaly_df, scalers)
#     anomaly_inputs, anomaly_targets = serials_to_samples(time_serials=normalized_anomaly_serials,
#                                                          window_length=window_length)
#
#     print(f'normalized_attack_serials shape: {normalized_attack_serials.shape}')
#     print(f'anomalies:')
#     print(anomalies)
#
#     print(f'model_name: {model_name}')
#     if not detect:
#         if epochs > 0:
#             evaluation_res = []
#             train_res = []
#             validate_res = []
#             for i in range(epochs):
#                 h = model.fit(train_inputs, train_targets, validation_split=0.1, batch_size=batch_size, epochs=1,
#                               shuffle=True, verbose=0)
#                 eval_loss = model.evaluate(test_inputs, test_targets, verbose=0)
#                 train_loss = h.history['loss'][-1]
#                 vali_loss = h.history['val_loss'][-1]
#                 evaluation_res.append(eval_loss)
#                 train_res.append(train_loss)
#                 validate_res.append(vali_loss)
#                 print(f"{i} - train_loss:{train_loss},val_loss:{vali_loss},eval_loss:{eval_loss}")
#             model.save_weights(model_weight_path)
#
#             history_csv_path = os.path.join(ROOT_DIR, "results", 'testbed', scenario, f'history-{model_name}.csv')
#             train_res = np.array(train_res).reshape([-1, 1])
#             validate_res = np.array(validate_res).reshape([-1, 1])
#             evaluation_res = np.array(evaluation_res).reshape([-1, 1])
#             records = pd.DataFrame(np.hstack([train_res, validate_res, evaluation_res]),
#                                    columns=['train_loss', 'val_loss', 'eval_loss'])
#
#             if os.path.exists(history_csv_path):
#                 history = pd.read_csv(history_csv_path)
#                 records = pd.concat([history, records], ignore_index=True)
#             print(f'loss records of {model_name}')
#             records.to_csv(history_csv_path, index=False)
#
#         if epochs >= 0:
#             outputs = []
#             for i in range(0, test_inputs.shape[0], 100):
#                 outputs.append(model(test_inputs[i:i + 100, :, :]))
#             outputs = np.vstack(outputs)
#             train_losses = np.abs(outputs - test_targets)
#             df = pd.DataFrame(data=train_losses, columns=remain_features)
#             df.to_csv(os.path.join(ROOT_DIR, "results", 'testbed', scenario, f'loss-{model_name}.csv'), index=False)
#
#             res = model.evaluate(test_inputs, test_targets)
#             print(res)
#
#             print_loss(model_name, train_losses, remain_features,
#                        os.path.join(ROOT_DIR, "results", 'testbed', scenario))
#
#         # return
#
#     threshold_csv_path = os.path.join(ROOT_DIR, 'results', 'testbed', scenario, f'loss-{model_name}.csv')
#     df = pd.read_csv(threshold_csv_path)
#     thresholds = define_threshold(df)
#
#     outputs = []
#     for i in range(0, attack_inputs.shape[0], 100):
#         outputs.append(model(attack_inputs[i:i + 100, :, :]))
#     outputs = np.vstack(outputs)
#     outputs = np.vstack((normalized_attack_serials[:window_length + jump + 1], outputs))
#     print(f'outputs shape: {outputs.shape}')
#
#     losses = np.abs(normalized_attack_serials - outputs)
#     print(f'losses shape: {outputs.shape}')
#
#     losses_copy = np.copy(losses)
#     # print(f'losses_copy shape: {losses_copy.shape}')
#     avg_losses = []
#     for i in range(losses_copy.shape[1]):
#         a_loss = losses_copy[:, i].reshape(-1)
#         # print(a_loss.shape)
#         a_avg_loss = np.convolve(a_loss, np.ones(avg_wl) / avg_wl, mode='valid')
#         # print(a_avg_loss.shape)
#         avg_losses.append(a_avg_loss.reshape([-1, 1]))
#
#     avg_losses = np.hstack(avg_losses)
#     avg_losses = np.vstack((np.zeros([avg_wl - 1, avg_losses.shape[1]]).astype(bool), avg_losses))
#     print(avg_losses.shape)
#
#     results = compare_with_threshold(avg_losses, thresholds)
#     # results = compare_with_threshold_max(avg_losses, thresholds)
#
#     print(f'results shape: {results.shape}')
#     # print(np.all(results==False))
#     spans = detect_results_to_spans(results)  # shape: [feature_size, threshold_size, ...]
#     print(spans)
#
#     threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per, threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration = span_analyze(
#         spans, anomalies, normalized_attack_serials.shape[0])
#
#     path = os.path.join(ROOT_DIR, "results", 'testbed', scenario, f'attack-{model_name}')
#     os.makedirs(path, exist_ok=True)
#     print_all(normal_serials=normalized_attack_serials, anomaly_serials=normalized_attack_serials, anomalies=anomalies,
#               outputs=outputs, feature_names=remain_features, spans=spans,
#               path=path)
#
#     return f'{"Bi" if bidirection else "Uni"} {f"{attn_layer}lyr" if attention else "No"}-Attn', threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per, \
#            threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, normal_duration

wl = 50
epo = 40

train_model(window_length=wl, epochs=epo, cell_num=48, dense_dim=48, bidirection=False, attn_layer=0)
# detect_anomalies(window_length=wl, cell_num=48, dense_dim=48, bidirection=False, attn_layer=0)

# train_model(window_length=wl, epochs=epo, cell_num=48, dense_dim=48, bidirection=False, attn_layer=1)
# detect_anomalies(window_length=wl, cell_num=48, dense_dim=48, bidirection=False, attn_layer=1)

# train_model(window_length=wl, epochs=epo, cell_num=48, dense_dim=48, bidirection=True, attn_layer=0)
# detect_anomalies(window_length=wl, cell_num=48, dense_dim=48, bidirection=True, attn_layer=0)

# train_model(window_length=wl, epochs=epo, cell_num=48, dense_dim=48, bidirection=True, attn_layer=1)
# detect_anomalies(window_length=wl, cell_num=48, dense_dim=48, bidirection=True, attn_layer=1)

# train_model(window_length=wl, epochs=epo, cell_num=48, dense_dim=48, bidirection=False, attn_layer=4)
# detect_anomalies(window_length=wl, cell_num=48, dense_dim=48, bidirection=False, attn_layer=4)

# train_model(window_length=wl, epochs=epo, cell_num=48, dense_dim=48, bidirection=True, attn_layer=4)
# detect_anomalies(window_length=wl, cell_num=48, dense_dim=48, bidirection=True, attn_layer=4)

wl = 100
epo = 40

# train_model(window_length=wl, epochs=epo, cell_num=48, dense_dim=48, bidirection=False, attn_layer=0)
# detect_anomalies(window_length=wl, cell_num=48, dense_dim=48, bidirection=False, attn_layer=0)

# train_model(window_length=wl, epochs=epo, cell_num=48, dense_dim=48, bidirection=False, attn_layer=1)
# detect_anomalies(window_length=wl, cell_num=48, dense_dim=48, bidirection=False, attn_layer=1)
