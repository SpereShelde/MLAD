import math
import os
from random import random, randint, choice
import numpy as np
import matplotlib.pyplot as plt
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from tools.DatasetReader import DatasetReader


def define_bias(partial_inputs):
    the_min = np.min(partial_inputs)
    the_max = np.max(partial_inputs)
    # the_range = the_max - the_min
    max_bias = 0.5 * the_max
    min_bias = 0.5 * the_min

    direction = choice([-1, 1])

    if direction > 0:
        return random() * (max_bias - min_bias) + min_bias
    else:
        return -1 * (random() * (0.5 * the_min) + the_min)


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

def insert_super_anomalies(serials, max_anom_duration, cooldown):
    #serials shape: [timesteps, features]
    time_steps = serials.shape[0]
    feature_size = serials.shape[1]
    anomalies = []
    anomaly_serials = np.copy(serials)

    current_feature = randint(0, feature_size)
    print(f'feature: {current_feature}')
    while current_feature < feature_size and current_feature >= 0:
        current_step = randint(max_anom_duration, math.floor(time_steps / 4))
        while current_step < time_steps - max_anom_duration:
            anomaly_type = choice(['bias', 'delay', 'replay'])
            if anomaly_type == 'bias':
                duration = randint(1, max_anom_duration)
                bias = define_bias(serials[max(0, current_step - duration):min(current_step + 2 * duration, time_steps - max_anom_duration), current_feature])
                if bias > 0:
                    anomaly_serials[current_step:current_step+duration, current_feature] = serials[current_step:current_step + duration, current_feature] + bias
                    anomalies.append([current_step, duration, current_feature, 'bias', bias])
            elif anomaly_type == 'delay':
                duration = randint(10, max_anom_duration)
                range = np.max(serials[current_step:current_step + duration, current_feature]) - np.min(serials[current_step:current_step + duration, current_feature])
                if range > 0:
                    delay = randint(math.floor(0.2 * duration), math.ceil(0.5 * duration))
                    anomaly_serials[current_step:current_step+delay, current_feature] = serials[current_step, current_feature]
                    anomaly_serials[current_step+delay:current_step+duration, current_feature] = serials[current_step:current_step + duration - delay, current_feature]
                    anomalies.append([current_step, duration, current_feature, 'delay', delay])
            elif anomaly_type == 'replay':
                duration = randint(10, max_anom_duration)
                replay = randint(math.floor(0.1 * duration), math.ceil(0.4 * duration))
                replay_inputs = serials[current_step - replay:current_step, current_feature]
                range = np.max(replay_inputs) - np.min(replay_inputs)
                if range > 0:
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
            current_step += max(cooldown, randint(0, math.floor(time_steps / 4)))
        current_feature += choice([-1, 1]) * randint(1, feature_size+1)
        print(f'feature: {current_feature}')

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
            ax[row, col].plot(x[ploted:to_plot], outputs[ploted:to_plot, i], label='predict', c='y', marker='.')

            for anomaly in anomalies:
                if anomaly[2] != i:
                    continue
                if anomaly[0]>=ploted and anomaly[0]+anomaly[1]<=to_plot:
                    ax[row, col].plot(x[anomaly[0]:anomaly[0]+anomaly[1]], anomaly_serials[anomaly[0]:anomaly[0] + anomaly[1], i], label=f'{anomaly[3]} attack', c='r', marker='.')
                elif anomaly[0]>=ploted and anomaly[0]<=to_plot and anomaly[0]+anomaly[1]>=to_plot:
                    ax[row, col].plot(x[anomaly[0]:to_plot], anomaly_serials[anomaly[0]:to_plot, i], label=f'{anomaly[3]} attack', c='r', marker='.')
                    anomaly[1] -= (to_plot - anomaly[0])
                    anomaly[0] = to_plot
                else:
                    pass
            ax[row, col].legend(loc="upper right")

            # colors = ['#E30000', '#F50067', '#D119BE', '#5B66FD', '#0089FF', '#0098E8']
            colors = ['r', 'orange', 'y', 'g', 'b', 'purple']
            alphas = [0.6, 0.55, 0.5, 0.45, 0.4, 0.35]
            for j in range(threshold_num):
                try:
                    for span in spans[i][j]:  # shape: [1, window_len, 1]
                        # print(span)
                        if span[0]>=ploted and span[1]<=to_plot:
                            ax[row, col].axvspan(span[0], span[1], facecolor=colors[j], alpha=alphas[j])
                        elif span[0]>=ploted and span[0]<=to_plot and span[1]>=to_plot:
                            ax[row, col].axvspan(span[0], to_plot, facecolor=colors[j], alpha=alphas[j])
                            span[0] = to_plot
                        else:
                            pass
                except TypeError as e:
                    print(i, j)
                    print(spans[i])

            ax[row, col].set_title(f'{feature_names[i]}')
            ax[row, col].set_ylim([0,1])
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
