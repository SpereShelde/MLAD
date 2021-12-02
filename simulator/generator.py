import math
import random

from matplotlib.patches import Patch

from settings import RLC_circuit, vehicle_turning, aircraft_pitch, dc_motor_position, quadrotor
from PID import PID
from control.matlab import lsim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# exp = RLC_circuit
# exp = vehicle_turning
exp = aircraft_pitch
# exp = dc_motor_position
# exp = quadrotor

# load model
dt = exp.Ts
sys = exp.sysc
pid = PID()
pid_att = PID()
pid.setWindup(100)
pid_att.setWindup(100)
pid.setSampleTime(dt)
pid_att.setSampleTime(dt)
ref = exp.ref
t_arr = exp.t_arr
# print(sys)

# init pid
pid.clear()
pid.setKp(exp.p)
pid.setKi(exp.i)
pid.setKd(exp.d)
pid.last_time = -1 * dt
pid.setControlLimit(exp.control_limit['lo'], exp.control_limit['up'])

pid_att.clear()
pid_att.setKp(exp.p)
pid_att.setKi(exp.i)
pid_att.setKd(exp.d)
pid_att.last_time = -1 * dt
pid_att.setControlLimit(exp.control_limit['lo'], exp.control_limit['up'])

# u
cin_arr = []
cin_arr_att = []
# y
y_real = exp.y_0
y_real_att = exp.y_0
y_real_arr = []
y_real_arr_att = []
y_measure = 0
y_measure_att = 0
y_measure_arr = []
y_measure_arr_att = []
# x
x_0 = exp.x_0
x_0_att = exp.x_0
x_real_arr = []
x_real_arr_att = []
x_measure = None
x_measure_att = None
x_measure_arr = []
x_measure_arr_att = []
x_measure_unatt_arr = []
x_measure_unatt_arr_att = []

x_prediction = None
x_prediction_att = None
x_prediction_arr = []
x_prediction_arr_att = []

att_starts = exp.attacks['starts']
att_ends = exp.attacks['ends']
att_durations = exp.attacks['durations']
att_types = exp.attacks['types']
att_values = exp.attacks['values']

end_of_att = False
att_idx = 0
current_att = [att_starts[att_idx], att_ends[att_idx], att_durations[att_idx], att_types[att_idx], att_values[att_idx]]
att_idx += 1

for i in range(0, exp.slot + 1):
    # for i in range(0, 3):
    if exp.y_index:
        y_real = y_real[exp.y_index]
        y_real_att = y_real_att[exp.y_index]
    y_real_arr.append(y_real)
    y_real_arr_att.append(y_real_att)
    x_real_arr.append(x_0.copy())  # shape (2,)
    x_real_arr_att.append(x_0_att.copy())  # shape (2,)

    # add noise
    # print(x_0, type(x_0))
    n = exp.sysc.A.shape[0]
    # how to choose noise
    # https: // mathworld.wolfram.com / BallPointPicking.html
    X_temp = np.random.standard_normal(n)
    Y_temp = np.random.exponential()
    X_scale = np.sqrt(Y_temp + np.inner(X_temp, X_temp))
    X_point = X_temp / X_scale
    noise = X_point * exp.epsilon

    x_measure = x_0 + noise
    x_measure_att = x_0_att + noise
    x_measure_unatt_arr.append(x_measure.copy())
    x_measure_unatt_arr_att.append(x_measure_att.copy())

    # attack!
    if not end_of_att:
        if current_att[0] <= i <= current_att[1]:
            att_type = current_att[-2]
            att_value = current_att[-1]
            if att_type == 0:
                x_measure_att[exp.x_index] += att_value
                x_measure_att[exp.x_index] += att_value
            elif att_type == 1:
                # att_value: delay steps
                if i - current_att[0] > att_value:
                    x_measure_att[exp.x_index] = x_measure_unatt_arr_att[i-att_value][exp.x_index]
                else:
                    x_measure_att[exp.x_index] = x_measure_unatt_arr_att[current_att[0]-1][exp.x_index]
            elif att_type == 2:
                j = i-current_att[0]
                x_measure_att[exp.x_index] = x_measure_unatt_arr_att[current_att[0]-att_value+j][exp.x_index]

        if current_att[1] < i:
            current_att = [att_starts[att_idx], att_ends[att_idx], att_durations[att_idx], att_types[att_idx],
                           att_values[att_idx]]
            att_idx += 1
            if att_idx == len(att_types):
                end_of_att = True

    if exp.y_index:
        y_measure = (np.asarray(exp.sysc.C) @ x_measure)[exp.y_index]
        y_measure_att = (np.asarray(exp.sysc.C) @ x_measure_att)[exp.y_index]
    else:
        y_measure = (np.asarray(exp.sysc.C) @ x_measure)[0]
        y_measure_att = (np.asarray(exp.sysc.C) @ x_measure_att)[0]
    y_measure_arr.append(y_measure)
    y_measure_arr_att.append(y_measure_att)
    x_measure_arr.append(x_measure)
    x_measure_arr_att.append(x_measure_att)

    # control or recovery
    pid.SetPoint = ref[i]
    pid_att.SetPoint = ref[i]
    pid.update(feedback_value=y_measure, current_time=i * dt)
    pid_att.update(feedback_value=y_measure_att, current_time=i * dt)
    cin = pid.output
    cin_att = pid_att.output
    cin_arr.append(cin)
    cin_arr_att.append(cin_att)
    # print(sys)
    yout, T, xout = lsim(sys, cin, [0, dt], x_0)
    yout_att, T_att, xout_att = lsim(sys, cin_att, [0, dt], x_0_att)
    # print('yout=', yout)
    y_real = yout[-1]
    y_rea_att = yout_att[-1]
    if len(xout.shape) == 1:
        x_0 = xout[-1]
        x_0_att = xout_att[-1]
    else:
        x_0 = xout[-1, :].T
        x_0_att = xout_att[-1, :].T


x_measure_arr = np.array(x_measure_arr)
x_measure_arr_att = np.array(x_measure_arr_att)
x_measure_unatt_arr_att = np.array(x_measure_unatt_arr_att)
ref = np.array(ref).reshape([-1, 1])
cin_arr = np.array(cin_arr).reshape([-1, 1])
cin_arr_att = np.array(cin_arr_att).reshape([-1, 1])

benign = np.hstack([ref, x_measure_arr, cin_arr])
anomalous = np.hstack([ref, x_measure_arr_att, cin_arr_att])
benign_pd = pd.DataFrame(benign)
anomalous_pd = pd.DataFrame(anomalous)

cols = ['ref'] + [f'x{i+1}' for i in range(len(exp.x_0))] + ['cin']
benign_pd.to_csv(f'benign-{exp.seed}.csv', columns=cols, index=False)
anomalous_pd.to_csv(f'anomalous-{exp.seed}.csv', columns=cols, index=False)
exit(0)

if exp.sep_graph:
    len_total = exp.one_graph_length * 3
    # len_total = x_measure_arr.shape[0]
    ploted = 0
    while ploted < len_total:
        to_plot = min(ploted + exp.one_graph_length, len_total)
        fig, ax = plt.subplots(nrows=exp.subfig_shape[0], ncols=exp.subfig_shape[1], figsize=(30, 16))

        for i in range(x_measure_arr.shape[1]):
            row = math.floor(i / exp.subfig_shape[1])
            col = i % exp.subfig_shape[1]
            if exp.subfig_shape[1] == 1:
                ax[row].plot(x_measure_arr_att[ploted:to_plot, i], label='anomalous', c='r', marker='.')
                ax[row].plot(x_measure_unatt_arr_att[ploted:to_plot, i], label='benign', c='b', marker='.')
                # ax[row].plot(x_measure_arr[ploted:to_plot, i], label='benign', c='b', marker='.')
            else:
                ax[row, col].plot(x_measure_arr_att[ploted:to_plot, i], label='anomalous', c='r', marker='.')
                # ax[row, col].plot(x_measure_arr[ploted:to_plot, i], label='benign', c='b', marker='.')
                ax[row].plot(x_measure_unatt_arr_att[ploted:to_plot, i], label='benign', c='b', marker='.')
        ax[-1].plot(ref[ploted:to_plot], label='benign', c='g', marker='.')

        legend_elements = []
        ax.legend(handles=legend_elements, loc='right', fontsize=12)
        for i in range(len(att_types)):
            start = att_starts[i]
            end = att_ends[i]
            type = att_types[i]
            value = att_values[i]
            if ploted <= start <= end <= to_plot:
                legend_elements.append(Patch(facecolor='y', edgecolor='y', label=f'{type}-{value:0.2f}'))
                if exp.subfig_shape[1] == 1:
                    ax[0].axvspan(start, end, facecolor='y', alpha=0.5)
                    ax[1].axvspan(start, end, facecolor='y', alpha=0.5)
                    ax[2].axvspan(start, end, facecolor='y', alpha=0.5)
                else:
                    pass


        ploted = to_plot
        plt.show()

