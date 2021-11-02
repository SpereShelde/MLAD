import matplotlib.pyplot as plt
import os
import csv
import pandas as pd

df = pd.read_csv('testbed_data\\complete.csv')

# df['Throttle_Control'][:1000].plot(color='red')
df = df.drop(['Servo', 'Servo_Control', 'Voltage', 'Throttle_Control', 'Throttle_A', 'Throttle_B', 'Acceleration_Z'], axis=1)
df[1000:1100].plot(figsize=(20, 6))
plt.show()