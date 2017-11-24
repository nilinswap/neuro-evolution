import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)
y = np.sin(x)

plt.ion()
ax = plt.gca()
ax.set_autoscale_on(True)
line, = ax.plot(x, y)

for i in range(100):
    line.set_ydata(y)
    ax.relim()
    ax.autoscale_view(True,True,True)
    plt.draw()
    y=y*1.1
    plt.pause(0.1)