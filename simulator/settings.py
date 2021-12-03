import random

from control.matlab import ss, linspace, c2d
from functools import partial
import numpy as np
import math


class Exp:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'control_limit', 'y_index', 'slot', 't_arr', 'attacks',
                 'sep_graph', 'epsilon', 'threshold', 'safeset', 'y_label',
                 'detection_window_1', 'detection_window_2', 'detection_window_init', 'x_index', 'seed', 'subfig_shape', 'one_graph_length'
                 ]

    def __init__(self, sysc, Ts):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.y_index = None  # used for system with multi-output


# ------- Vehicle Turing -------------

# model
A = [[-25 / 3]]
B = [[5]]
C = [[1]]
D = [[0]]
sysc = ss(A, B, C, D)
dt = 0.02
vehicle_turning = Exp(sysc, dt)
exp = vehicle_turning
exp.name = 'VehicleTurning'
exp.x_0 = np.array([0])
exp.y_0 = 0
exp.control_limit = {'lo': np.array([-3]), 'up': np.array([3])}
# exp.epsilon = 0.05
exp.epsilon = 0.075
exp.safeset = {'lo': np.array([-2]), 'up': np.array([2])}
exp.x_index = 0  # change which state

# control
exp.p = 0.5
exp.i = 7
exp.d = 0

# time
t_total = 10
exp.slot = int(t_total / dt)
exp.t_arr = linspace(0, t_total, exp.slot + 1)

# reference
exp.ref = [0] * 251 + [0.5] * 250

# attacks
exp.attacks = {'modification': {}, 'delay': {}, 'replay': {}}
exp.attacks['modification']['func'] = partial((lambda x, y: x - y), y=0.5)
exp.attacks['modification']['start'] = int(7 / dt)
exp.attacks['modification']['end'] = int(7.5 / dt)
exp.attacks['delay']['step'] = 60
exp.attacks['delay']['start'] = int(5 / dt)
exp.attacks['delay']['end'] = int(7.5 / dt)
exp.attacks['replay']['first'] = int(5 / dt)
exp.attacks['replay']['start'] = int(6 / dt)
exp.attacks['replay']['end'] = int(3.5 / dt)

# detection
exp.threshold = np.array([0.07])
exp.detection_window_1 = 4
exp.detection_window_2 = 40
exp.detection_window_init = 10

# graph
exp.sep_graph = True
exp.attacks['modification']['xlim'] = (6.8, 7.4)
exp.attacks['modification']['ylim'] = (0.4, 0.95)
exp.attacks['modification']['seed'] = 33  # 26
exp.attacks['delay']['xlim'] = (4.8, 5.4)
exp.attacks['delay']['ylim'] = (-0.1, 0.8)
exp.attacks['delay']['seed'] = 31
exp.attacks['replay']['xlim'] = (5.4, 6.45)
exp.attacks['replay']['ylim'] = (0.4, 0.95)
exp.attacks['replay']['seed'] = 54  # 29
exp.y_label = 'Speed Difference'

# ------- RLC Circuit -------------

# model
R = 10000
L = 0.5
C = 0.0001

A = [[0, 1 / C], [-1 / L, -R / L]]
B = [[0], [1 / L]]
C = [[1, 0]]
D = [[0]]
sysc = ss(A, B, C, D)
dt = 0.02
RLC_circuit = Exp(sysc, dt)
exp = RLC_circuit
exp.name = 'RLCCircuit'
exp.x_0 = np.array([0, 0])
exp.y_0 = 0
exp.control_limit = {'lo': np.array([-5]), 'up': np.array([5])}
exp.epsilon = 1.7e-2
exp.safeset = {'lo': np.array([-3.5, -5]), 'up': np.array([3.5, 5])}
exp.x_index = 0  # change which state

# control
exp.p = 5
exp.i = 5
exp.d = 0

# time
t_total = 10
exp.slot = int(t_total / dt)
exp.t_arr = linspace(0, t_total, exp.slot + 1)

# reference
exp.ref = [1] * 201 + [2.5] * 200 + [2.5] * 100

# attacks
exp.attacks = {'modification': {}, 'delay': {}, 'replay': {}}
exp.attacks['modification']['func'] = partial((lambda x, y: x - y), y=1)
exp.attacks['modification']['start'] = int(5 / dt)
exp.attacks['modification']['end'] = int(6.5 / dt)
exp.attacks['delay']['step'] = 50
exp.attacks['delay']['start'] = int(4.42 / dt)
exp.attacks['delay']['end'] = int(3.5 / dt)
exp.attacks['replay']['first'] = int(4.16/dt)
exp.attacks['replay']['start'] = int(5 / dt)
exp.attacks['replay']['end'] = int(3.5 / dt)

# detection
exp.threshold = np.array([0.04, 0.01])  # 7e-5
exp.detection_window_1 = 4
exp.detection_window_2 = 40
exp.detection_window_init = 10

# graph
exp.sep_graph = True
exp.attacks['modification']['xlim'] = (4.4, 5.5)
exp.attacks['modification']['ylim'] = (2, 3.8)
exp.attacks['modification']['seed'] = 5
exp.attacks['delay']['xlim'] = (4.1, 5.25)
exp.attacks['delay']['ylim'] = (1, 4)
exp.attacks['delay']['seed'] = 8  # until 52
exp.attacks['replay']['xlim'] = (4.7, 5.45)
exp.attacks['replay']['ylim'] = (2.25, 3.5)
exp.attacks['replay']['seed'] = 4 #4,8, until 51
exp.y_label = 'Capacitor Voltage'

# ------- Aircraft Pitch -------------
seed = 14
random.seed(seed)
np.random.seed(seed)

# model
A = [[-0.313, 56.7, 0],
     [-0.0139, -0.426, 0],
     [0, 56.7, 0]]
B = [[0.232], [0.0203], [0]]
C = [[0, 0, 1]]
D = [[0]]
sysc = ss(A, B, C, D)
dt = 0.05
aircraft_pitch = Exp(sysc, dt)
exp = aircraft_pitch
exp.seed = seed
exp.name = 'AircraftPitch'
exp.x_0 = np.array([0, 0, 0])
exp.y_0 = 0

control_lo = -7
control_up = 7
exp.control_limit = {'lo': np.array([control_lo]), 'up': np.array([control_up])}
exp.epsilon = 1e-17
exp.safeset = {'lo': np.array([-np.inf, -np.inf, -2.5]), 'up': np.array([np.inf, np.inf, 2.5])}
exp.x_index = 2   # anomalous index

# control
exp.p = 14
exp.i = 0.8
exp.d = 5.7

# time
t_total = 500
exp.slot = int(t_total / dt)
exp.t_arr = linspace(0, t_total, exp.slot + 1)

# reference
exp.ref = []
steps = int(t_total / dt) + 1
while len(exp.ref) < steps:
    interval = random.randint(100, 400)
    reference = random.random() * (control_up - control_lo) * 0.2 + control_lo * 0.5
    exp.ref += [reference] * interval

exp.ref = exp.ref[:steps]

attack_intervals = np.random.poisson(150, math.floor(steps / 150))
attack_starts = np.cumsum(attack_intervals)
attack_durations = np.random.poisson(50, len(attack_intervals))
attack_ends = attack_starts + attack_durations

attack_ends = attack_ends[attack_ends < steps-50]
attack_durations = attack_durations[:len(attack_ends)]
attack_starts = attack_starts[:len(attack_ends)]
attack_types = np.random.choice([0, 1, 2], len(attack_ends))
attack_values = []

bias_up = 0.2
bias_lo = 0.1
for i, att in enumerate(attack_types):
    if att == 0:
        bias = random.random() * (bias_up - bias_lo) + bias_lo
        direction = random.choice([-1,1])
        attack_values.append(direction*bias)
    elif att == 1:
        duration = attack_durations[i]
        delay = random.randint(int(duration*0.1), int(duration*0.6))
        attack_values.append(delay)
    else:
        duration = attack_durations[i]
        replay = random.randint(int(duration*0.1), int(duration*0.3))
        attack_values.append(replay)

exp.attacks = {'starts': attack_starts, 'ends': attack_ends, 'durations': attack_durations, 'types': attack_types, 'values': attack_values}

# graph
exp.sep_graph = True
exp.subfig_shape = (3, 1)
exp.one_graph_length = 400
exp.y_label = 'Pitch'


# ------- DC Motor Position -------------

# model
J = 0.01
b = 0.1
K = 0.01
R = 1
L = 0.5

A = [[0, 1, 0],
     [0, -b / J, K / J],
     [0, -K / L, -R / L]]
B = [[0], [0], [1 / L]]
C = [[1, 0, 0]]
D = [[0]]
sysc = ss(A, B, C, D)
dt = 0.1
dc_motor_position = Exp(sysc, dt)
exp = dc_motor_position
exp.name = 'DCMotorPosition'
exp.x_0 = np.array([0, 0, 0])
exp.y_0 = 0
exp.control_limit = {'lo': np.array([-20]), 'up': np.array([20])}
exp.epsilon = 0
exp.safeset = {'lo': np.array([-4, -np.inf, -np.inf]), 'up': np.array([4, np.inf, np.inf])}
exp.x_index = 0

# control
exp.p = 11
exp.i = 0
exp.d = 5

# time
t_total = 12
exp.slot = int(t_total / dt)
exp.t_arr = linspace(0, t_total, exp.slot + 1)

# reference
exp.ref = [math.pi / 2] * 41 + [0] * 80

# detection
exp.threshold = np.array([1e-1, 1, 1])
exp.detection_window_1 = 4
exp.detection_window_2 = 100
exp.detection_window_init = 10

# attacks
exp.attacks = {'modification': {}, 'delay': {}, 'replay': {}}
exp.attacks['modification']['func'] = partial((lambda x, y: x - y), y=2)
exp.attacks['modification']['start'] = int(6 / dt)
exp.attacks['modification']['end'] = int(7.5 / dt)
exp.attacks['delay']['step'] = 60
exp.attacks['delay']['start'] = int(5 / dt)
exp.attacks['delay']['end'] = int(7.5 / dt)
exp.attacks['replay']['first'] = int(5 / dt)
exp.attacks['replay']['start'] = int(6 / dt)
exp.attacks['replay']['end'] = int(3.5 / dt)

# graph
exp.sep_graph = True
# exp.attacks['modification']['xlim'] = (6.8, 7.4)
# exp.attacks['modification']['ylim'] = (0, 1.5)
exp.attacks['modification']['seed'] = 0
# exp.attacks['delay']['xlim'] = (4.8, 5.4)
# exp.attacks['delay']['ylim'] = (-0.1, 0.8)
# exp.attacks['delay']['seed'] = 31
# exp.attacks['replay']['xlim'] = (5.4, 6.45)
# exp.attacks['replay']['ylim'] = (0.4, 0.95)
# exp.attacks['replay']['seed'] = 54  # 29
exp.y_label = 'Rotation Angle'

# ------- Quadrotor -------------

# model
g = 9.81
m = 0.468
A = [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, -g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
B = [[0], [0], [0], [0], [0], [0], [0], [0], [1 / m], [0], [0], [0]]
C = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
D = [[0], [0], [0], [0], [0], [0]]
sysc = ss(A, B, C, D)
dt = 0.1
quadrotor = Exp(sysc, dt)
exp = quadrotor
exp.name = 'Quadrotor'
exp.x_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
exp.y_0 = np.array([0, 0, 0, 0, 0, 0])
exp.y_index = 5
exp.control_limit = {'lo': np.array([-2]), 'up': np.array([2])}
exp.epsilon = 1.56e-15
exp.safeset = {'lo': np.array([-np.inf]*8+[-5]), 'up': np.array([np.inf]*8+[5])}
exp.x_index = 8

# control
exp.p = 0.8 #0.1
exp.i = 0
exp.d = 1

# time
t_total = 30
exp.slot = int(t_total / dt)
exp.t_arr = linspace(0, t_total, exp.slot + 1)

# reference
# exp.ref = [2] * 401 + [4] * 800 + [4] * 300
exp.ref = [2] * 101 + [4] * 200

# detection
exp.threshold = np.array([1, 1, 1, 1, 1, 1, 1, 1, 6e-16])
exp.detection_window_1 = 10
exp.detection_window_2 = 40
exp.detection_window_init = 10

# attacks
exp.attacks = {'modification': {}, 'delay': {}, 'replay': {}}
exp.attacks['modification']['func'] = partial((lambda x, y: x - y), y=6)
exp.attacks['modification']['start'] = int(15 / dt)
exp.attacks['modification']['end'] = int(7.5 / dt)
exp.attacks['delay']['step'] = 100
exp.attacks['delay']['start'] = int(5 / dt)
exp.attacks['delay']['end'] = int(7.5 / dt)
exp.attacks['replay']['first'] = int(5 / dt)
exp.attacks['replay']['start'] = int(6 / dt)
exp.attacks['replay']['end'] = int(3.5 / dt)

# graph
exp.sep_graph = True
# exp.attacks['modification']['xlim'] = (11, 17)
# exp.attacks['modification']['ylim'] = (1, 8)
exp.attacks['modification']['seed'] = 0
# exp.attacks['delay']['xlim'] = (4.8, 5.4)
# exp.attacks['delay']['ylim'] = (-0.1, 0.8)
# exp.attacks['delay']['seed'] = 31
# exp.attacks['replay']['xlim'] = (5.4, 6.45)
# exp.attacks['replay']['ylim'] = (0.4, 0.95)
# exp.attacks['replay']['seed'] = 54  # 29
exp.y_label = 'Altitude'
