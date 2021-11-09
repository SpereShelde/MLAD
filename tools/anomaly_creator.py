import math
import os
from random import random, randint, choice
import numpy as np
import matplotlib.pyplot as plt
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from tools.DatasetReader import DatasetReader


def define_bias(partial_inputs):
    # partial_inputs: only one feature
    # shape: [3 * attack_duration, 1]
    # given partial inputs before and after the attack duration
    # computes variance and max value
    # returns a random value between 0 and the smaller one between 1/2 of range and 1/10 of max value
    range = np.max(partial_inputs) - np.min(partial_inputs)
    the_max = np.max(partial_inputs)
    return choice([-1, 1]) * random() * min(0.5 * range, 0.1 * the_max)

def insert_anomalies(inputs, max_anom_duration, cooldown):
    #inputs shape: [timesteps, features]
    time_steps = inputs.shape[0]
    feature_size = inputs.shape[1]
    anomalies = []
    anomaly_inputs = np.zeros(inputs.shape)
    current_step = randint(max_anom_duration, time_steps - max_anom_duration)

    while current_step < time_steps - max_anom_duration:
        duration = randint(1, max_anom_duration)
        current_feature = randint(0, feature_size)
        while current_feature < feature_size and current_feature >= 0:
            anomaly_type = choice(['bias', 'delay', 'replay'])
            # anomaly_type = 'bias'
            if anomaly_type == 'bias':
                bias = define_bias(inputs[max(0, current_step-duration):min(current_step+2*duration, time_steps - max_anom_duration), current_feature])
                if bias > 0:
                    anomaly_inputs[current_step:current_step+duration, current_feature] = inputs[current_step:current_step+duration, current_feature] + bias
                    anomalies.append((current_step, duration, current_feature, 'bias', bias))
            elif anomaly_type == 'delay':
                range = np.max(inputs[current_step:current_step+duration, current_feature]) - np.min(inputs[current_step:current_step+duration, current_feature])
                if range > 0:
                    delay = randint(math.floor(0.2 * duration), math.ceil(0.5 * duration))
                    anomaly_inputs[current_step:current_step+delay, current_feature] = inputs[current_step, current_feature]
                    anomaly_inputs[current_step+delay:current_step+duration, current_feature] = inputs[current_step:current_step+duration-delay, current_feature]
                    anomalies.append((current_step, duration, current_feature, 'delay', delay))
            elif anomaly_type == 'replay':
                replay = randint(math.floor(0.1 * duration), math.ceil(0.4 * duration))
                replay_inputs = inputs[current_step-replay:current_step, current_feature]
                range = np.max(replay_inputs[current_step:current_step+duration, current_feature]) - np.min(replay_inputs[current_step:current_step+duration, current_feature])
                if range > 0:
                    replay_step = current_step + replay
                    while replay_step < current_step + duration:
                        anomaly_inputs[replay_step-replay:replay_step, current_feature] = replay_inputs
                        replay_step += replay
                    anomaly_inputs[replay_step-replay:current_step + duration, current_feature] = replay_inputs[:current_step + duration - replay_step+replay]
                    anomalies.append((current_step, duration, current_feature, 'replay', replay))
            else:
                print("?")
                exit(0)
            current_feature += choice([-1, 1]) * randint(1, feature_size+1)

        current_step += duration
        current_step += max(cooldown, randint(0, time_steps))
    return anomaly_inputs, anomalies

def insert_super_anomalies(inputs, max_anom_duration, cooldown):
    #inputs shape: [timesteps, features]
    time_steps = inputs.shape[0]
    feature_size = inputs.shape[1]
    anomalies = []
    anomaly_inputs = np.zeros(inputs.shape)

    current_feature = randint(0, feature_size)
    # print(f'feature: {current_feature}')
    while current_feature < feature_size and current_feature >= 0:
        current_step = randint(max_anom_duration, time_steps - max_anom_duration)
        while current_step < time_steps - max_anom_duration:
            anomaly_type = choice(['bias', 'delay', 'replay'])
            if anomaly_type == 'bias':
                duration = randint(1, max_anom_duration)
                bias = define_bias(inputs[max(0, current_step-duration):min(current_step+2*duration, time_steps - max_anom_duration), current_feature])
                if bias > 0:
                    anomaly_inputs[current_step:current_step+duration, current_feature] = inputs[current_step:current_step+duration, current_feature] + bias
                    anomalies.append((current_step, duration, current_feature, 'bias', bias))
            elif anomaly_type == 'delay':
                duration = randint(10, max_anom_duration)
                range = np.max(inputs[current_step:current_step+duration, current_feature]) - np.min(inputs[current_step:current_step+duration, current_feature])
                if range > 0:
                    delay = randint(math.floor(0.2 * duration), math.ceil(0.5 * duration))
                    anomaly_inputs[current_step:current_step+delay, current_feature] = inputs[current_step, current_feature]
                    anomaly_inputs[current_step+delay:current_step+duration, current_feature] = inputs[current_step:current_step+duration-delay, current_feature]
                    anomalies.append((current_step, duration, current_feature, 'delay', delay))
            elif anomaly_type == 'replay':
                duration = randint(10, max_anom_duration)
                replay = randint(math.floor(0.1 * duration), math.ceil(0.4 * duration))
                replay_inputs = inputs[current_step-replay:current_step, current_feature]
                range = np.max(replay_inputs) - np.min(replay_inputs)
                if range > 0:
                    replay_step = current_step + replay
                    while replay_step < current_step + duration:
                        anomaly_inputs[replay_step-replay:replay_step, current_feature] = replay_inputs
                        replay_step += replay
                    anomaly_inputs[replay_step-replay:current_step + duration, current_feature] = replay_inputs[:current_step + duration - replay_step+replay]
                    anomalies.append((current_step, duration, current_feature, 'replay', replay))
            else:
                print("?")
                exit(0)

            current_step += duration
            current_step += max(cooldown, randint(0, time_steps))
        current_feature += choice([-1, 1]) * randint(1, feature_size+1)
        # print(f'feature: {current_feature}')

    return anomaly_inputs, anomalies

def print_all(inputs, anomaly_inputs, anomalies, feature_names, len_one_graph=500, len_total=2000):
    len_total = inputs.shape[0]
    x = np.arange(len_total).reshape(-1, 1)
    feature_size = inputs.shape[1]
    ploted = 0
    while ploted < len_total:
        plt.figure(figsize=(30, 16))
        to_plot = min(ploted + len_one_graph, len_total)
        if not any([anomaly[0]>=ploted and anomaly[0]+anomaly[1]<=to_plot for anomaly in anomalies]):
            ploted = to_plot
            continue

        for i in range(feature_size):
            plt.subplot(math.ceil(feature_size / 2), 2, i + 1)
            plt.plot(x[ploted:to_plot], inputs[ploted:to_plot, i], label='origin', c='b', marker='.')
            # plt.plot(x[ploted:to_plot], anomaly_inputs[ploted:to_plot, i], label='ano', c='r', marker='.')
            for anomaly in anomalies:
                if anomaly[2] == i and anomaly[0]>=ploted and anomaly[0]+anomaly[1]<=to_plot:
                    plt.plot(x[anomaly[0]:anomaly[0]+anomaly[1]], anomaly_inputs[anomaly[0]:anomaly[0]+anomaly[1], i], label=f'{anomaly[3]} attack', c='r', marker='.')
                    plt.legend(loc="upper right")

            plt.title(f'{feature_names[i]}')
        plt.savefig(os.path.join(ROOT_DIR, "attack", f'{ploted}-{to_plot}.png'))
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
    print_all(inputs=time_serials, anomaly_inputs=anomaly_inputs, anomalies=anomalies, feature_names=selected_features)

test(time_serials)
