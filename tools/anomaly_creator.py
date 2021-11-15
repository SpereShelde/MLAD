import math
import os
from random import random, randint, choice
import numpy as np
import matplotlib.pyplot as plt
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from tools.DatasetReader import DatasetReader

avg_anomaly_happen_interval = 300

def define_bias(partial_inputs, global_max, global_min):
    ratio_max = 0.5
    ratio_min = 0.25
    if global_max - global_min < 2:
        ratio_max = 0.9
        ratio_min = 0.6
    the_min = np.min(partial_inputs)
    the_max = np.max(partial_inputs)
    pos_bias = global_max - the_max
    neg_bias = the_min - global_min
    # direction = choice([-1, 1])

    if pos_bias > neg_bias:
        bias = random() * (pos_bias * ratio_max) + pos_bias * ratio_min
    else:
        bias = -1 * (random() * (neg_bias * ratio_max) + neg_bias * ratio_min)

    # print(f'global_max:{global_max}, the_min:{the_min}, pos_bias:{pos_bias}, neg_bias:{neg_bias}, bias:{bias}')
    return bias

# def define_bias_nomalized(partial_inputs):
#     the_max = partial_inputs.max()
#     the_min = partial_inputs.min()
#     if the_min < 0 and the_max > 1:
#         return 0
#     ratio_min = 0.4
#     ratio_max = 0.8
#     ratio_range = ratio_max - ratio_min
#     if the_min <= 0.5 and the_max >= 0.5:
#         if the_min < 0:
#             direction = 1
#         elif the_max > 1:
#             direction = -1
#         else:
#             direction = choice([-1, 1])
#         if direction == 1:
#             max_bias = 1-the_max
#         else:
#             max_bias = the_min
#         bias = direction * (random() * (ratio_range * max_bias) + ratio_min * max_bias)
#     elif the_min >= 0.5 and the_max >= 0.5:
#         max_bias = the_min
#         bias = -1 * (random() * (ratio_range * max_bias) + ratio_min * max_bias)
#     elif the_max <= 0.5 and the_max <= 0.5:
#         max_bias = 1-the_max
#         bias = random() * (ratio_range * max_bias) + ratio_min * max_bias
#     print(partial_inputs)
#     print(f'min:{the_min}, max:{the_max}, bias: {bias}')
#     return bias

def sample_from_np(np_inputs, window_length=50):
    input_time_serials = []
    size = np_inputs.shape[0]
    for i in range(size - window_length):
        a_sample = np_inputs[i:i + window_length]
        input_time_serials.append(a_sample)
    return np.array(input_time_serials)


def detect_results_to_spans(results):
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
                if stop >= start:
                    spans[j][i].append([start, stop])
                start = stop
    return spans

def compare_with_threshold(outputs, thresholds):
    results = []
    for threshold in thresholds:
        results.append(np.less_equal(outputs, threshold))
    results = np.array(results)
    return results

def insert_super_anomalies(serials, feature_ids=None, max_anom_duration=50, cooldown=50):
    #serials shape: [timesteps, features]
    time_steps = serials.shape[0]
    feature_size = serials.shape[1]
    anomalies = []
    anomaly_serials = np.copy(serials)
    feature_ids = list.copy(feature_ids)
    current_feature = choice(feature_ids)
    # feature_ids.remove(current_feature)
    attack_feature_num = choice([1,2,3])
    # print(serials.shape)
    # exit(0)
    for i in range(attack_feature_num):
        print(current_feature)
        current_step = randint(max_anom_duration, avg_anomaly_happen_interval)
        while current_step < time_steps - max_anom_duration:
            # anomaly_type = choice(['bias', 'delay', 'replay'])
            anomaly_type = choice(['bias'])
            if anomaly_type == 'bias':
                duration = randint(1, max_anom_duration)
                global_max = serials[:, current_feature].max()
                global_min = serials[:, current_feature].min()
                bias = define_bias(serials[max(0, current_step - duration):min(current_step + 2 * duration, time_steps - max_anom_duration), current_feature], global_max, global_min)
                if bias != 0:
                    anomaly_serials[current_step:current_step+duration, current_feature] = serials[current_step:current_step + duration, current_feature] + bias
                    anomalies.append([current_step, duration, current_feature, 'bias', bias])
            elif anomaly_type == 'delay':
                duration = randint(10, max_anom_duration)
                the_range = np.max(serials[current_step:current_step + duration, current_feature]) - np.min(serials[current_step:current_step + duration, current_feature])
                if the_range > 0:
                    delay = randint(math.floor(0.2 * duration), math.ceil(0.5 * duration))
                    anomaly_serials[current_step:current_step+delay, current_feature] = serials[current_step, current_feature]
                    anomaly_serials[current_step+delay:current_step+duration, current_feature] = serials[current_step:current_step + duration - delay, current_feature]
                    anomalies.append([current_step, duration, current_feature, 'delay', delay])
            elif anomaly_type == 'replay':
                duration = randint(10, max_anom_duration)
                replay = randint(math.floor(0.1 * duration), math.ceil(0.4 * duration))
                replay_inputs = serials[current_step - replay:current_step, current_feature]
                the_range = np.max(replay_inputs) - np.min(replay_inputs)
                if the_range > 0:
                    replay_step = current_step + replay
                    while replay_step < current_step + duration:
                        anomaly_serials[replay_step-replay:replay_step, current_feature] = replay_inputs
                        replay_step += replay
                    anomaly_serials[replay_step-replay:current_step + duration, current_feature] = replay_inputs[:current_step + duration - replay_step+replay]
                    anomalies.append([current_step, duration, current_feature, 'replay', replay])
            else:
                print("?")
                exit(0)

            current_step += duration
            current_step += max(cooldown, randint(0, avg_anomaly_happen_interval))
        feature_ids.remove(current_feature)
        current_feature = choice(feature_ids)

    return anomaly_serials, anomalies


def print_all(normal_serials, anomaly_serials, anomalies, feature_names, len_one_graph=500, outputs=None, spans=None, path='.'):
    threshold_num = len(spans[0])
    len_total = normal_serials.shape[0]
    x = np.arange(len_total).reshape(-1, 1)
    feature_size =len(feature_names)
    ploted = 0
    while ploted < len_total:
        # plt.figure(figsize=(30, 16))
        to_plot = min(ploted + len_one_graph, len_total)
        # if not any([anomaly[0]>=ploted and anomaly[0]+anomaly[1]<=to_plot for anomaly in anomalies]):
        #     ploted = to_plot
        #     continue
        rows = math.ceil(feature_size / 2)
        fig, ax = plt.subplots(nrows=rows, ncols=2, figsize=(30, 16))

        for i in range(feature_size):
            row = math.floor(i / 2)
            col = i % 2
            ax[row, col].plot(x[ploted:to_plot], normal_serials[ploted:to_plot, i], label='origin', c='b', marker='.')
            ax[row, col].plot(x[ploted:to_plot], outputs[ploted:to_plot, i], label='predict', c='g', marker='.')

            for anomaly in anomalies:
                if anomaly[2] != i:
                    continue
                if anomaly[0]>=ploted and anomaly[0]+anomaly[1]<=to_plot:
                    ax[row, col].plot(x[anomaly[0]:anomaly[0]+anomaly[1]], anomaly_serials[anomaly[0]:anomaly[0] + anomaly[1], i], label=f'{anomaly[3]} atk:{anomaly[4]:.2f}', c='r', marker='.')
                elif anomaly[0]>=ploted and anomaly[0]<=to_plot and anomaly[0]+anomaly[1]>=to_plot:
                    ax[row, col].plot(x[anomaly[0]:to_plot], anomaly_serials[anomaly[0]:to_plot, i], label=f'{anomaly[3]} atk:{anomaly[4]:.2f}', c='r', marker='.')
                    anomaly[1] -= (to_plot - anomaly[0])
                    anomaly[0] = to_plot
                else:
                    pass
            ax[row, col].legend(loc="upper right")

            # colors = ['#E30000', '#F50067', '#D119BE', '#5B66FD', '#0089FF', '#0098E8']
            # colors = ['r', 'orange', 'y', 'g', 'b', 'purple']
            # alphas = [0.6, 0.55, 0.5, 0.45, 0.4, 0.35]
            for j in range(threshold_num-1,0,-1):
                try:
                    for span in spans[i][j]:  # shape: [1, window_len, 1]
                        # print(span)
                        if span[0]>=ploted and span[1]<=to_plot:
                            ax[row, col].axvspan(span[0], span[1], facecolor='y', alpha=0.2)
                        elif span[0]>=ploted and span[0]<=to_plot and span[1]>=to_plot:
                            ax[row, col].axvspan(span[0], to_plot, facecolor='y', alpha=0.2)
                            span[0] = to_plot
                        else:
                            pass
                except TypeError as e:
                    print(i, j)
                    print(spans[i])

            ax[row, col].set_title(f'{feature_names[i]}')
            ax[row, col].set_ylim([-1,2])
        # fig.tight_layout()
        plt.savefig(os.path.join(path, f'{ploted}-{to_plot}.png'))
        plt.close()
        ploted = to_plot

selected_features = ['/CAN/Yawrate1', '/CAN/ENG_Trq_DMD', '/CAN/VehicleSpeed', '/CAN/AccPedal', '/CAN/ENG_Trq_ZWR',
                 '/Plugins/Velocity_X', '/GPS/Direction', '/CAN/EngineTemperature']
files = ["20181117_Driver1_Trip7.hdf"]

data_reader = DatasetReader(
        [
            "20181113_Driver1_Trip1.hdf", "20181113_Driver1_Trip2.hdf", "20181116_Driver1_Trip3.hdf",
            "20181116_Driver1_Trip4.hdf", "20181116_Driver1_Trip5.hdf", "20181116_Driver1_Trip6.hdf",
            "20181117_Driver1_Trip7.hdf", "20181117_Driver1_Trip8.hdf", "20181203_Driver1_Trip9.hdf",
            "20181203_Driver1_Trip10.hdf",
        ])

time_serials, _ = data_reader._concatenate_data(file_names=files, feature_names=selected_features)

def test(time_serials):
    # print(time_serials.shape)
    # anomaly_inputs, anomalies = insert_anomalies(time_serials, 50, 50)
    anomaly_inputs, anomalies = insert_super_anomalies(time_serials, 50, 50)
    while len(anomalies) == 0:
        anomaly_inputs, anomalies = insert_super_anomalies(time_serials, 50, 50)
    print(anomalies)
    print_all(normal_serials=time_serials, anomaly_serials=anomaly_inputs, anomalies=anomalies, feature_names=selected_features)

# test(time_serials)
