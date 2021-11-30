import random
from settings import RLC_circuit, vehicle_turning, aircraft_pitch, dc_motor_position, quadrotor
from PID import PID
from control.matlab import lsim
import matplotlib.pyplot as plt
import numpy as np

# exp = RLC_circuit
# exp = vehicle_turning
exp = aircraft_pitch
# exp = dc_motor_position
# exp = quadrotor
attack = 'modification'
# attack = 'delay'
# attack = 'replay'

# load model
dt = exp.Ts
sys = exp.sysc
pid = PID()
pid.setWindup(100)
pid.setSampleTime(dt)
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

# u
cin_arr = []
# y
y_real = exp.y_0
y_real_arr = []
y_measure = 0
y_measure_arr = []
# x
x_0 = exp.x_0
x_real_arr = []
x_measure = None
x_measure_arr = []
x_measure_unatt_arr = []

x_prediction = None
x_prediction_arr = []

att = exp.attacks[attack]
t_att_start = att['start']
t_att_end = att['end']

for i in range(0, exp.slot + 1):
    # for i in range(0, 3):
    if exp.y_index:
        y_real = y_real[exp.y_index]
    y_real_arr.append(y_real)
    x_real_arr.append(x_0.copy())  # shape (2,)

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
    x_measure_unatt_arr.append(x_measure.copy())

    # attack!
    if t_att_start <= i:
        if attack == 'modification':
            x_measure[exp.x_index] = att['func'](x_measure[exp.x_index])
        elif attack == 'delay':
            if i - t_att_start > att['step']:
                x_measure[exp.x_index] = x_measure_unatt_arr[i-att['step']][exp.x_index]
            else:
                x_measure[exp.x_index] = x_measure_unatt_arr[t_att_start-1][exp.x_index]
        elif attack == 'replay':
            j = i-t_att_start
            # print(x_measure_unatt_arr[att['first']+j])
            x_measure[exp.x_index] = x_measure_unatt_arr[att['first']+j][exp.x_index]

    if exp.y_index:
        y_measure = (np.asarray(exp.sysc.C) @ x_measure)[exp.y_index]
    else:
        y_measure = (np.asarray(exp.sysc.C) @ x_measure)[0]
    y_measure_arr.append(y_measure)
    x_measure_arr.append(x_measure)

    # control or recovery
    pid.SetPoint = ref[i]
    pid.update(feedback_value=y_measure, current_time=i * dt)
    cin = pid.output
    cin_arr.append(cin)
    # print(sys)
    yout, T, xout = lsim(sys, cin, [0, dt], x_0)
    # print('yout=', yout)
    y_real = yout[-1]
    if len(xout.shape) == 1:
        x_0 = xout[-1]
    else:
        x_0 = xout[-1, :].T



if exp.sep_graph:
    plt.figure()
    # plt.plot(t_arr, y_measure_arr)
    plt.plot(t_arr, ref)
    plt.plot(t_arr, cin_arr)
    # plt.plot(t_arr, x_real_arr)
    # filename = os.path.join(rp, 'no_attack.jpg')
    # plt.savefig(filename)
    plt.show()
