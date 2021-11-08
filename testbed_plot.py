import math

import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(ROOT_DIR, 'data', 'testbed', 'barnes')

# testbed_files = [f for f in listdir(data_dir) if isfile(os.path.join(data_dir, f)) and f[-3:] == "csv" and f[:12] == "speed_attack"]
testbed_files = [f for f in listdir(data_dir) if isfile(os.path.join(data_dir, f)) and f[-3:] == "csv"]

for file in testbed_files:
    plt.figure(figsize=(30, 16))
    plt.suptitle(file, fontsize=30)

    df = pd.read_csv(os.path.join(data_dir, file))
    x = np.arange(df.shape[0]).reshape(-1, 1)
    i = 0
    col_size = len(df.columns)
    rows = math.ceil(col_size / 3)
    for col in df.columns:
        i += 1
        plt.subplot(rows, 3, i)
        plt.plot(x, df[col], c='b', marker='.')
        plt.title(col)
        plt.xlim(1500, 2000)
    plt.savefig(os.path.join(data_dir, f'{file[:-4]}.png'))
    # exit(0)
