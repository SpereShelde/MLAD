import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile
from simple_kf import SimpleKF

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(ROOT_DIR, 'data', 'testbed')

# name = 'speed_attack_Linear_X_1.csv'

attack_files = [f for f in listdir(data_dir) if isfile(os.path.join(data_dir, f)) and f[-3:] == "csv" and f[:12] == "speed_attack"]


def plot_one_feature(feature_name, idx):
    plt.subplot(5, 2, idx)
    plt.title(feature_name)

    plt.plot(plot_range_before, df[feature_name].iloc[plot_range_before], c='b', marker='.')
    plt.plot(bias_range, df[feature_name].iloc[bias_range], c='r', marker='.')
    plt.plot([plot_range_before[-1], bias_range[0]], [df[feature_name][plot_range_before[-1]], df[feature_name][bias_range[0]]], c='r', marker='.')
    if attack_sensor == feature_name:
        plt.plot([plot_range_before[-1], bias_range[0]], [df[feature_name][plot_range_before[-1]], df[feature_name][bias_range[0]] - df['Bias_val'][bias_range[0]]], c='b', marker='.')
        plt.plot(bias_range, df[feature_name].iloc[bias_range] - df['Bias_val'].iloc[bias_range], c='b', marker='.')
        plt.plot([bias_range[-1], plot_range_after[0]], [df[feature_name][bias_range[-1]] - df['Bias_val'][bias_range[-1]], df[feature_name][plot_range_after[0]]], c='b', marker='.')
    plt.plot([bias_range[-1], plot_range_after[0]], [df[feature_name][bias_range[-1]], df[feature_name][plot_range_after[0]]], c='r', marker='.')
    plt.plot(plot_range_after, df[feature_name].iloc[plot_range_after], c='b', marker='.')

for name in attack_files:

    df = pd.read_csv(os.path.join(data_dir, name))
    # df = df.drop(['Servo', 'Servo_Control', 'Voltage', 'Throttle_Control', 'Throttle_A', 'Throttle_B', 'Angular_Z', 'Linear_Y', 'Acceleration_Z'], axis=1)
    attack_sensor = df['Bias_Sensor'][1]
    bias_range = df.index[df['Bias_val'] != 0].tolist()
    bias_len = len(bias_range)
    # plot_range = [i for i in range(bias_range[0] - bias_len, bias_range[0])] + bias_range + [i for i in range(bias_range[-1], bias_range[-1] + bias_len)]
    plot_range_before = [i for i in range(bias_range[0] - bias_len, bias_range[0])]
    plot_range_after = [i for i in range(bias_range[-1] + 1, bias_range[-1] + bias_len + 1)]
    if plot_range_before[0] < 0:
        plot_range_after = plot_range_after + [i for i in range(plot_range_after[-1] + 1, plot_range_after[-1] - plot_range_before[0] + 1)]
        plot_range_before = [i for i in plot_range_before if i >= 0]
    # print(plot_range_before)
    # print(bias_range)
    # print(plot_range_after)
    # exit(0)

    plt.figure(figsize=(30, 16))
    plt.suptitle(name[:-4], fontsize=20)

    plot_one_feature('Linear_X', 1)

    kf_vel_z = SimpleKF(0.5, 0.5, 0.5)
    vel_z = df['Linear_Z']
    vel_z_kf = []
    for vz in vel_z:
        vel_z_kf.append(kf_vel_z.update_estimate(vz))
    df['Linear_Z'] = vel_z_kf
    plot_one_feature('Linear_Z', 2)

    kf_acc_x = SimpleKF(1.2, 1.2, 0.5)
    acc_x = df['Acceleration_X']
    acc_x_kf = []
    for accx in acc_x:
        acc_x_kf.append(kf_acc_x.update_estimate(accx))
    df['Acceleration_X'] = acc_x_kf
    plot_one_feature('Acceleration_X', 3)


    kf_acc_y = SimpleKF(1.2, 1.2, 0.5)
    acc_y = df['Acceleration_Y']
    acc_y_kf = []
    for accy in acc_y:
        acc_y_kf.append(kf_acc_y.update_estimate(accy))
    df['Acceleration_Y'] = acc_y_kf
    plot_one_feature('Acceleration_Y', 4)


    kf_acc_z = SimpleKF(2.4, 2.4, 0.5)
    acc_z = df['Acceleration_Z']
    acc_z_kf = []
    for accz in acc_z:
        acc_z_kf.append(kf_acc_z.update_estimate(accz))
    plt.subplot(4, 2, 5)
    df['Acceleration_Z'] = acc_z_kf
    plot_one_feature('Acceleration_Z', 5)


    kf_ang_x = SimpleKF(0.06, 0.06, 0.5)
    ang_x = df['Angular_X']
    ang_x_kf = []
    for angx in ang_x:
        ang_x_kf.append(kf_ang_x.update_estimate(angx))
    df['Angular_X'] = ang_x_kf
    plot_one_feature('Angular_X', 6)


    kf_ang_y = SimpleKF(0.06, 0.06, 0.5)
    ang_y = df['Angular_Y']
    ang_y_kf = []
    for angy in ang_y:
        ang_y_kf.append(kf_ang_y.update_estimate(angy))
    df['Angular_Y'] = ang_y_kf
    plot_one_feature('Angular_Y', 7)


    kf_ang_z = SimpleKF(0.05, 0.05, 0.5)
    ang_z = df['Angular_Z']
    ang_z_kf = []
    for angz in ang_z:
        ang_z_kf.append(kf_ang_z.update_estimate(angz))
    df['Angular_Z'] = ang_z_kf
    plot_one_feature('Angular_Z', 8)

    plot_one_feature('Reference', 9)
    plot_one_feature('Throttle_Control', 10)

    # plt.show()
    plt.savefig(os.path.join(ROOT_DIR, 'data', 'testbed', 'attack_plot', f'{name[:-4]}.png'))
