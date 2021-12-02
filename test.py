import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import pandas as pd


a = np.array([
        [1,2,3],
        [0,0,0],
    ])
a[:, 1] = 3
print(a)
