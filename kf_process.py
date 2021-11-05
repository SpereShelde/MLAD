import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'testbed', 'room_train.csv'))
# df = df.drop(['idx'], axis=1)

#
# # df['Throttle_Control'][:1000].plot(color='red')
# df = df.drop(['Servo', 'Servo_Control', 'Voltage', 'Throttle_Control', 'Throttle_A', 'Throttle_B', 'Angular_Z', 'Linear_Y', 'Acceleration_Z'], axis=1)
# df[1000:1100].plot(figsize=(20, 6))
# df.plot(figsize=(20, 6))
# plt.show()
# exit(0)

plt.figure(figsize=(30, 16))
x_range = [800, 900]
x_len = x_range[1] - x_range[0]
# x = range(x_len)
x = range(df.shape[0])

from simple_kf import SimpleKF

# kf_vel_x = SimpleKF(0.04, 0.04, 0.6)
# vel_x = df['Linear_X']
# vel_x_kf = []
# for vx in vel_x:
#     vel_x_kf.append(kf_vel_x.update_estimate(vx))
plt.subplot(4, 2, 1)
# plt.plot(x, vel_x, c='b', marker='.')
# df['Linear_X'] = vel_x_kf
plt.plot(x, df['Linear_X'], c='r', marker='.')
plt.xlim(x_range)

kf_vel_z = SimpleKF(0.5, 0.5, 0.5)
vel_z = df['Linear_Z']
vel_z_kf = []
for vz in vel_z:
    vel_z_kf.append(kf_vel_z.update_estimate(vz))
plt.subplot(4, 2, 2)
plt.plot(x, vel_z, c='b', marker='.')
df['Linear_Z'] = vel_z_kf
plt.plot(x, df['Linear_Z'], c='r', marker='.')
plt.xlim(x_range)

kf_acc_x = SimpleKF(1.2, 1.2, 0.5)
acc_x = df['Acceleration_X']
acc_x_kf = []
for accx in acc_x:
    acc_x_kf.append(kf_acc_x.update_estimate(accx))
plt.subplot(4, 2, 3)
plt.plot(x, acc_x, c='b', marker='.')
df['Acceleration_X'] = acc_x_kf
plt.plot(x, df['Acceleration_X'], c='r', marker='.')
plt.xlim(x_range)

kf_acc_y = SimpleKF(1.2, 1.2, 0.5)
acc_y = df['Acceleration_Y']
acc_y_kf = []
for accy in acc_y:
    acc_y_kf.append(kf_acc_y.update_estimate(accy))
plt.subplot(4, 2, 4)
plt.plot(x, acc_y, c='b', marker='.')
df['Acceleration_Y'] = acc_y_kf
plt.plot(x, df['Acceleration_Y'], c='r', marker='.')
plt.xlim(x_range)

kf_acc_z = SimpleKF(2.4, 2.4, 0.5)
acc_z = df['Acceleration_Z']
acc_z_kf = []
for accz in acc_z:
    acc_z_kf.append(kf_acc_z.update_estimate(accz))
plt.subplot(4, 2, 5)
plt.plot(x, acc_z, c='b', marker='.')
df['Acceleration_Z'] = acc_z_kf
plt.plot(x, df['Acceleration_Z'], c='r', marker='.')
plt.xlim(x_range)

kf_ang_x = SimpleKF(0.06, 0.06, 0.5)
ang_x = df['Angular_X']
ang_x_kf = []
for angx in ang_x:
    ang_x_kf.append(kf_ang_x.update_estimate(angx))
plt.subplot(4, 2, 6)
plt.plot(x, ang_x, c='b', marker='.')
df['Angular_X'] = ang_x_kf
plt.plot(x, df['Angular_X'], c='r', marker='.')
plt.xlim(x_range)

kf_ang_y = SimpleKF(0.06, 0.06, 0.5)
ang_y = df['Angular_Y']
ang_y_kf = []
for angy in ang_y:
    ang_y_kf.append(kf_ang_y.update_estimate(angy))
plt.subplot(4, 2, 7)
plt.plot(x, ang_y, c='b', marker='.')
df['Angular_Y'] = ang_y_kf
plt.plot(x, df['Angular_Y'], c='r', marker='.')
plt.xlim(x_range)

kf_ang_z = SimpleKF(0.05, 0.05, 0.5)
ang_z = df['Angular_Z']
ang_z_kf = []
for angz in ang_z:
    ang_z_kf.append(kf_ang_z.update_estimate(angz))
plt.subplot(4, 2, 8)
plt.plot(x, ang_z, c='b', marker='.')
df['Angular_Z'] = ang_z_kf
plt.plot(x, df['Angular_Z'], c='r', marker='.')
plt.xlim(x_range)

# plt.show()

df.to_csv('room_train_kf.csv')
