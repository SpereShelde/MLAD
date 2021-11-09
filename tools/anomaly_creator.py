import math
from random import random, randint, choice
from numpy import np

def define_bias(partial_inputs):
    # partial_inputs: only one feature
    # shape: [3 * attack_duration, 1]
    # given partial inputs before and after the attack duration
    # computes variance and max value
    # returns a random value between 0 and the smaller one between 1/2 of variance and 1/10 of max value
    var = np.var(partial_inputs)
    the_max = np.max(partial_inputs)
    return choice([-1, 1]) * random() * min(0.5 * var, 0.1 * the_max)

def inset_anomalies(inputs, max_anom_duration, cooldown):
    #inputs shape: [timesteps, features]
    time_steps = inputs.shape[0]
    feature_size = inputs.shape[1]
    anomalies = []
    anomaly_inputs = np.copy(inputs)
    current_step = randint(max_anom_duration, time_steps - max_anom_duration)
    duration = randint(1, max_anom_duration)
    while current_step < time_steps - max_anom_duration:
        current_feature = randint(0, feature_size)
        while current_feature < feature_size:
            # anomaly_type = choice(['bias', 'delay', 'replay'])
            anomaly_type = 'bias'
            if anomaly_type == 'bias':
                bias = define_bias(inputs[max(0, current_step-duration):min(current_step+2*duration, time_steps - max_anom_duration), current_feature])
                anomaly_inputs[current_step:current_step+duration, current_feature] += bias
                anomalies.append((current_step, duration, 'bias', bias))
            elif anomaly_type == 'delay':
                delay = randint(math.floor(0.2 * duration), math.ceil(0.5 * duration))
                anomaly_inputs[current_step:current_step+delay, current_feature] = inputs[current_step, current_feature]
                anomaly_inputs[current_step+delay:current_step+duration, current_feature] = inputs[current_step:current_step+duration-delay, current_feature]
                anomalies.append((current_step, duration, 'delay', delay))
            elif anomaly_type == 'replay':
                replay = randint(math.floor(0.1 * duration), math.ceil(0.4 * duration))
                replay_inputs = inputs[current_step-replay, current_step, current_feature]
                replay_step = current_step + replay
                while replay_step < current_step + duration:
                    anomaly_inputs[replay_step-replay:replay_step, current_feature] = replay_inputs
                    replay_step += replay
                anomaly_inputs[replay_step-replay:current_step + duration, current_feature] = replay_inputs[:current_step + duration - replay_step+replay]
                anomalies.append((current_step, duration, 'replay', replay))
            else:
                print("?")
                exit(0)
            current_feature += choice([-1, 1]) * randint(1, feature_size+1)

        current_step += duration
        current_step += max(cooldown, randint(0, time_steps))
    return inputs

def test():
    pass

