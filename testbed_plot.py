import matplotlib.pyplot as plt
import os
import csv
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'testbed', 'outside_at_end.csv'))

# df['Throttle_Control'][:1000].plot(color='red')
df = df.drop(['Servo', 'Servo_Control', 'Voltage', 'Throttle_Control', 'Throttle_A', 'Throttle_B', 'Acceleration_Z', 'Linear_Y'], axis=1)
df.plot(figsize=(20, 6))
plt.show()
