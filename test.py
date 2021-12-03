import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import pandas as pd

df = pd.read_csv('test.csv')

length = df.shape[0]
noise = np.random.normal(0, 0.01, length)

df['x1'] += noise
plt.plot(df['x1'])

plt.show()
