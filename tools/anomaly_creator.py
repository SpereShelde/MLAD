import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# from classic_rnn.Testbed_model import sample_from_csv
from tools.DatasetReader import DatasetReader
import pandas as pd

seed = 1200
# interval = 300
random.seed(seed)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# avg_anomaly_happen_interval = interval

def define_bias(partial_inputs, global_max, global_min):
    # ratio_max = 0.7
    # ratio_min = 0.4
    ratio_max = 0.6
    ratio_min = 0.2

    the_min = np.min(partial_inputs)
    the_max = np.max(partial_inputs)
    # if the_max - the_min < (global_max - global_min) / 10:
    #     ratio_max = 0.9
    #     ratio_min = 0.6

    pos_bias = global_max - the_max
    neg_bias = the_min - global_min

    if pos_bias > neg_bias:
        bias = random.random() * (pos_bias * ratio_max) + pos_bias * ratio_min
    else:
        bias = -1 * (random.random() * (neg_bias * ratio_max) + neg_bias * ratio_min)

    return bias


def sample_from_np(np_inputs, window_length=50):
    input_time_serials = []
    size = np_inputs.shape[0]
    for i in range(size - window_length - 1):
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


def compare_with_threshold(losses, thresholds):
    results = []
    for threshold in thresholds:
        # pad_falses = np.zeros([avg_wl-1, outputs.shape[1]]).astype(bool)
        # results.append(np.vstack((pad_falses, np.less_equal(outputs, threshold))))
        results.append(np.less_equal(losses, threshold))
    results = np.array(results)
    return results


def compare_with_threshold_max(losses, thresholds):
    results = []
    feature_size = losses.shape[1]
    for threshold in thresholds:
        result = np.less_equal(losses, threshold)
        result_num = np.count_nonzero(result, axis=1)
        for (i, num) in enumerate(result_num):
            if num <= feature_size - 2:
                result[i] = [True] * feature_size
                result[i, np.argmax(losses[i])] = False
        results.append(result)
    results = np.array(results)
    return results


def insert_super_anomalies(serials, feature_ids=None, max_anom_duration=50, cooldown=50, window_len=50,
                           avg_anomaly_interval=300):
    # serials shape: [timesteps, features]
    time_steps = serials.shape[0]
    feature_size = serials.shape[1]
    anomalies = []
    anomaly_serials = np.copy(serials)
    feature_ids = list.copy(feature_ids)
    current_feature = random.choice(feature_ids)
    # feature_ids.remove(current_feature)
    # attack_feature_num = 1
    attack_feature_num = random.choice([1, 2])
    # print(serials.shape)
    for i in range(attack_feature_num):
        global_var = np.var(serials[:, current_feature])
        global_max = serials[:, current_feature].max()
        global_min = serials[:, current_feature].min()
        # print(f'global_var, {global_var}')
        current_step = random.randint(window_len, window_len + avg_anomaly_interval)
        while current_step < time_steps - max_anom_duration:
            anomaly_type = random.choice(['bias', 'delay', 'replay'])
            # anomaly_type = random.choice(['bias'])
            duration = random.randint(10, max_anom_duration)
            if anomaly_type == 'bias':
                bias = define_bias(serials[max(0, current_step - duration):min(current_step + 2 * duration,
                                                                               time_steps - max_anom_duration),
                                   current_feature], global_max, global_min)
                if bias != 0:
                    anomaly_serials[current_step:current_step + duration, current_feature] = serials[
                                                                                             current_step:current_step + duration,
                                                                                             current_feature] + bias
                    anomalies.append([current_step, duration, current_feature, 'bias', bias])
            elif anomaly_type == 'delay':
                # the_range = np.max(serials[current_step:current_step + duration, current_feature]) - np.min(serials[current_step:current_step + duration, current_feature])
                the_var = np.var(serials[current_step:current_step + duration, current_feature])
                # print(f'delay, var:{the_var}')
                if the_var > global_var / 10:
                    delay = random.randint(math.floor(0.2 * duration), math.ceil(0.5 * duration))
                    anomaly_serials[current_step:current_step + delay, current_feature] = serials[
                        current_step, current_feature]
                    anomaly_serials[current_step + delay:current_step + duration, current_feature] = serials[
                                                                                                     current_step:current_step + duration - delay,
                                                                                                     current_feature]
                    anomalies.append([current_step, duration, current_feature, 'delay', delay])
            elif anomaly_type == 'replay':
                replay = random.randint(math.floor(0.1 * duration), math.ceil(0.4 * duration))
                replay_inputs = serials[current_step - replay:current_step, current_feature]
                the_var = np.var(replay_inputs)
                # print(f'replay, var:{the_var}')
                # the_range = np.max(replay_inputs) - np.min(replay_inputs)
                if the_var > global_var / 10:
                    replay_step = current_step + replay
                    while replay_step < current_step + duration:
                        anomaly_serials[replay_step - replay:replay_step, current_feature] = replay_inputs
                        replay_step += replay
                    anomaly_serials[replay_step - replay:current_step + duration, current_feature] = replay_inputs[
                                                                                                     :current_step + duration - replay_step + replay]
                    anomalies.append([current_step, duration, current_feature, 'replay', replay])
            else:
                print("?")
                exit(0)

            current_step += duration
            current_step += random.randint(cooldown, avg_anomaly_interval)
        feature_ids.remove(current_feature)
        current_feature = random.choice(feature_ids)

    return anomaly_serials, anomalies


def span_analyze(spans, anomalies, total_duration):
    # span shape: [feature, threshold, alert_durations]
    # anomalies: a list of [current_step, duration, current_feature, 'bias', bias]
    anomalies_copy = list.copy(anomalies)
    spans_copy = list.copy(spans)

    feature_size = len(spans_copy)
    threshold_size = len(spans_copy[0])
    anomaly_size = len(anomalies_copy)

    # detected_anomalies = [0] * threshold_size
    threshold_anomaly_detect_duration = [0] * threshold_size
    threshold_anomaly_detect_delay = [0] * threshold_size
    threshold_anomaly_tp = [0] * threshold_size
    anomalous_duration = 0

    for anomaly in anomalies_copy:
        anomalous_feature = anomaly[2]
        anomaly_start = anomaly[0]
        anomaly_end = anomaly[0] + anomaly[1] - 1

        anomalous_duration += anomaly[1]
        anomaly.append(False)  # denotes anomaly not detected by default
        for j in range(threshold_size):
            for a_span in spans_copy[anomalous_feature][j]:
                if len(a_span) == 2:
                    a_span.append(False)
                    a_span.append(-1)
                span_start = a_span[0]
                span_end = a_span[1] - 1

                if span_start < anomaly_start < span_end:
                    threshold_anomaly_detect_duration[j] += (min(anomaly_end, span_end) - anomaly_start)
                    if not anomaly[-1]:
                        threshold_anomaly_tp[j] += 1
                        anomaly[-1] = True
                        # detected_anomalies[j] += 1
                        # threshold_anomaly_detect_delay[j] += 0

                    # a_span[-2] = 'TP'
                    a_span[-1] = anomaly_start - span_start

                elif anomaly_start <= span_start <= anomaly_end:
                    threshold_anomaly_detect_duration[j] += (min(anomaly_end, span_end) - span_start)
                    if not anomaly[-1]:
                        threshold_anomaly_tp[j] += 1
                        anomaly[-1] = True
                        # detected_anomalies[j] += 1
                        threshold_anomaly_detect_delay[j] += (span_start - anomaly_start)

                    a_span[-2] = True
                    # a_span[-1] = span_start - anomaly_start
                    # anomaly[-2] = 'det'
                    # anomaly[-1] +=

                else:
                    a_span[-1] = span_end - span_start

    threshold_FP_duration = [0] * threshold_size
    for i in range(feature_size):
        for j in range(threshold_size):
            for a_span in spans_copy[i][j]:
                if len(a_span) == 4 and not a_span[-2]:
                    threshold_FP_duration[j] += a_span[-1]

    threshold_anomaly_detect_duration = np.array(threshold_anomaly_detect_duration)
    threshold_anomaly_detect_delay = np.array(threshold_anomaly_detect_delay)
    threshold_anomaly_tp = np.array(threshold_anomaly_tp)
    # detected_anomalies = np.array(detected_anomalies)
    threshold_FP_duration = np.array(threshold_FP_duration)

    threshold_anomaly_detect_duration_per = threshold_anomaly_detect_duration / anomalous_duration
    threshold_anomaly_TP_rate = threshold_anomaly_tp / anomaly_size

    threshold_FP_per = threshold_FP_duration / (total_duration - anomalous_duration)

    threshold_anomaly_detect_delay_avg = []
    for i in range(threshold_size):
        delay = threshold_anomaly_detect_delay[i]
        tp = threshold_anomaly_tp[i]
        if tp == 0:
            threshold_anomaly_detect_delay_avg.append(0)
        else:
            threshold_anomaly_detect_delay_avg.append(delay / tp)
    threshold_anomaly_detect_delay_avg = np.array(threshold_anomaly_detect_delay_avg)

    return threshold_anomaly_TP_rate, threshold_anomaly_detect_duration_per, threshold_anomaly_detect_delay_avg, threshold_FP_per, anomaly_size, anomalous_duration, (
                total_duration - anomalous_duration)


def print_all(normal_serials, anomaly_serials, anomalies, feature_names, len_one_graph=500, outputs=None, spans=None,
              path='.'):
    threshold_num = len(spans[0])
    len_total = normal_serials.shape[0]
    x = np.arange(len_total).reshape(-1, 1)
    feature_size = len(feature_names)
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
                if anomaly[0] >= ploted and anomaly[0] + anomaly[1] <= to_plot:
                    ax[row, col].plot(x[anomaly[0]:anomaly[0] + anomaly[1]],
                                      anomaly_serials[anomaly[0]:anomaly[0] + anomaly[1], i],
                                      label=f'{anomaly[3]} atk:{anomaly[4]:.2f}', c='r', marker='.')
                elif anomaly[0] >= ploted and anomaly[0] <= to_plot and anomaly[0] + anomaly[1] >= to_plot:
                    ax[row, col].plot(x[anomaly[0]:to_plot], anomaly_serials[anomaly[0]:to_plot, i],
                                      label=f'{anomaly[3]} atk:{anomaly[4]:.2f}', c='r', marker='.')
                    anomaly[1] -= (to_plot - anomaly[0])
                    anomaly[0] = to_plot
                else:
                    pass
            ax[row, col].legend(loc="upper right")

            lower_ys = [-1, -0.4, 0.2, 0.8, 1.4]
            upper_ys = [-0.4, 0.2, 0.8, 1.4, 2]
            alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
            for j in range(threshold_num):
                try:
                    for span in spans[i][j]:  # shape: [1, window_len, 1]
                        # print(span)
                        if span[0] >= ploted and span[1] <= to_plot:
                            ax[row, col].axvspan(span[0], span[1], lower_ys[j], upper_ys[j], facecolor='y',
                                                 alpha=alphas[j])
                        elif span[0] >= ploted and span[0] <= to_plot and span[1] >= to_plot:
                            ax[row, col].axvspan(span[0], to_plot, lower_ys[j], upper_ys[j], facecolor='y',
                                                 alpha=alphas[j])
                            span[0] = to_plot
                        else:
                            pass
                except TypeError as e:
                    print(i, j)
                    print(spans[i])

            ax[row, col].set_title(f'{feature_names[i]}')
            ax[row, col].set_ylim([-1, 2])
        # fig.tight_layout()
        plt.savefig(os.path.join(path, f'{ploted}-{to_plot}.png'))
        plt.close()
        ploted = to_plot


sub_corelated_features = ['/CAN/AccPedal', '/CAN/ENG_Trq_ZWR', '/CAN/ENG_Trq_m_ex',
                          '/CAN/EngineSpeed_CAN', '/CAN/Trq_Indicated', '/CAN/VehicleSpeed',
                          '/CAN/WheelSpeed_FL', '/CAN/WheelSpeed_FR', '/CAN/WheelSpeed_RL', '/CAN/WheelSpeed_RR']


def test_testbed():
    df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'data', 'simulator', 'aircraft_pitch_test.csv'))
    anomaly_serials, anomalies = insert_super_anomalies(df.to_numpy(), feature_ids=[2,3,4,5],
                                                        max_anom_duration=50, cooldown=50, window_len=50,
                                                        avg_anomaly_interval=300)

    print(f'anomaly_serials shape: {anomaly_serials.shape}')
    print('anomalies:')
    print(anomalies)

    df = pd.DataFrame(anomaly_serials, columns=['Reference', 'Control', 'Measure', 'State_1', 'State_2', 'State_3'])
    df.to_csv(os.path.join(ROOT_DIR, '..', 'data', 'simulator', 'aircraft_pitch_Anomalous_Serials.csv'), index=False)
    df = pd.DataFrame(anomalies, columns=['Start', 'Duration', 'Feature_id', 'Type', 'Value'])
    df.to_csv(os.path.join(ROOT_DIR, '..', 'data', 'simulator', 'aircraft_pitch_Anomalies.csv'), index=False)

def testbed_plot():
    serials_df = pd.read_csv(os.path.join(ROOT_DIR, 'aircraft_pitch_Anomalous_Serials.csv'))
    anomaly_serials = serials_df.to_numpy(dtype=float)
    anomalies_df = pd.read_csv(os.path.join(ROOT_DIR, 'aircraft_pitch_Anomalies.csv'))
    anomalies = anomalies_df.values.tolist()

    # len_total = anomaly_serials.shape[0]
    len_total = 5000
    x = np.arange(len_total).reshape(-1, 1)
    feature_size = 4
    ploted = 0
    while ploted < len_total:
        to_plot = min(ploted + 500, len_total)
        rows = math.ceil(feature_size / 2)
        fig, ax = plt.subplots(nrows=rows, ncols=2, figsize=(30, 16))

        for i in range(feature_size):
            row = math.floor(i / 2)
            col = i % 2
            ax[row, col].plot(x[ploted:to_plot], anomaly_serials[ploted:to_plot, i], label='origin', c='b', marker='.')

            for anomaly in anomalies:
                if anomaly[2] != i:
                    continue
                if anomaly[0] >= ploted and anomaly[0] + anomaly[1] <= to_plot:
                    ax[row, col].plot(x[anomaly[0]:anomaly[0] + anomaly[1]],
                                      anomaly_serials[anomaly[0]:anomaly[0] + anomaly[1], i],
                                      label=f'{anomaly[3]} atk:{anomaly[4]:.2f}', c='r', marker='.')
                elif anomaly[0] >= ploted and anomaly[0] <= to_plot and anomaly[0] + anomaly[1] >= to_plot:
                    ax[row, col].plot(x[anomaly[0]:to_plot], anomaly_serials[anomaly[0]:to_plot, i],
                                      label=f'{anomaly[3]} atk:{anomaly[4]:.2f}', c='r', marker='.')
                    anomaly[1] -= (to_plot - anomaly[0])
                    anomaly[0] = to_plot
                else:
                    pass
            ax[row, col].legend(loc="upper right")

            ax[row, col].set_title(f'{sub_corelated_features[i]}')
            # ax[row, col].set_ylim([-1, 2])
        # fig.tight_layout()
        plt.savefig(f'{ploted}-{to_plot}.png')
        plt.close()
        ploted = to_plot

def test():
    data_reader = DatasetReader(["20181203_Driver1_Trip10.hdf"])

    time_serials, _ = data_reader._concatenate_data(file_names=["20181203_Driver1_Trip10.hdf"],
                                                    feature_names=sub_corelated_features)

    attack_ids = [i for i in range(len(sub_corelated_features))]
    anomaly_serials, anomalies = insert_super_anomalies(time_serials, feature_ids=attack_ids, max_anom_duration=100,
                                                        cooldown=100, window_len=90,
                                                        avg_anomaly_interval=400)
    print(f'anomaly_serials shape: {anomaly_serials.shape}')
    print('anomalies:')
    print(anomalies)
    df = pd.DataFrame(anomaly_serials, columns=sub_corelated_features)
    df.to_csv('20181203_Driver1_Trip10_Anomalous_Serials.csv', index=False)
    df = pd.DataFrame(anomalies, columns=['Start', 'Duration', 'Feature_id', 'Type', 'Value'])
    df.to_csv('20181203_Driver1_Trip10_Anomalies.csv', index=False)


def test_plot():
    serials_df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'classic_rnn', "results", 'sub_corelated_features',
                                          '20181203_Driver1_Trip10_Anomalous_Serials.csv'))
    anomaly_serials = serials_df.to_numpy(dtype=float)
    anomalies_df = pd.read_csv(os.path.join(ROOT_DIR, '..', 'classic_rnn', "results", 'sub_corelated_features',
                                            '20181203_Driver1_Trip10_Anomalies.csv'))
    anomalies = anomalies_df.values.tolist()

    # len_total = anomaly_serials.shape[0]
    len_total = 2000
    x = np.arange(len_total).reshape(-1, 1)
    feature_size = len(sub_corelated_features)
    ploted = 0
    while ploted < len_total:
        to_plot = min(ploted + 500, len_total)
        rows = math.ceil(feature_size / 2)
        fig, ax = plt.subplots(nrows=rows, ncols=2, figsize=(30, 16))

        for i in range(feature_size):
            row = math.floor(i / 2)
            col = i % 2
            ax[row, col].plot(x[ploted:to_plot], anomaly_serials[ploted:to_plot, i], label='origin', c='b', marker='.')

            for anomaly in anomalies:
                if anomaly[2] != i:
                    continue
                if anomaly[0] >= ploted and anomaly[0] + anomaly[1] <= to_plot:
                    ax[row, col].plot(x[anomaly[0]:anomaly[0] + anomaly[1]],
                                      anomaly_serials[anomaly[0]:anomaly[0] + anomaly[1], i],
                                      label=f'{anomaly[3]} atk:{anomaly[4]:.2f}', c='r', marker='.')
                elif anomaly[0] >= ploted and anomaly[0] <= to_plot and anomaly[0] + anomaly[1] >= to_plot:
                    ax[row, col].plot(x[anomaly[0]:to_plot], anomaly_serials[anomaly[0]:to_plot, i],
                                      label=f'{anomaly[3]} atk:{anomaly[4]:.2f}', c='r', marker='.')
                    anomaly[1] -= (to_plot - anomaly[0])
                    anomaly[0] = to_plot
                else:
                    pass
            ax[row, col].legend(loc="upper right")

            ax[row, col].set_title(f'{sub_corelated_features[i]}')
            # ax[row, col].set_ylim([-1, 2])
        # fig.tight_layout()
        plt.savefig(f'{ploted}-{to_plot}.png')
        plt.close()
        ploted = to_plot

# test_plot()
test_testbed()
# testbed_plot()