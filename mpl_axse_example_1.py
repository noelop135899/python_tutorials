import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


fig = plt.figure()

x = np.arange(0,10,0.1)
y1 = 0.05*x**2
y2 = -1*y1

ax1 = plt.subplot()
ax2 = ax1.twinx()
ax1.plot(x,y1,'r')
ax2.plot(x,y2,'b')

ax3 = fig.add_axes([0.2,0.4,0.25,0.25])
ax3.plot(x,y1,'r')

ax1.set_xlabel('Label X')
ax1.set_ylabel('Label Y')
ax2.set_ylabel('Label twinx Y2')
ax3.set_title('AX1')

plt.show()