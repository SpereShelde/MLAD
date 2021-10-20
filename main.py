import numpy as np
import matplotlib.pyplot as plt


x = np.arange(200)
output = np.arange(200)
targets_test = np.arange(200)


plt.plot(x, output, label='predict', color='r', marker='+')
plt.plot(x, targets_test, label='target', color='b', marker='x')

plt.xlabel('idx')
plt.ylabel('value')
plt.title('Result')
plt.legend()
plt.show()
