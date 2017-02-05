import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3,3,50)
y1 = 2*x + 1
y2 = x**2

plt.figure()

plt.xlim(-2,2)
plt.ylim(-2,2)

new_ticks = np.linspace(-2,2,9)
plt.xticks(new_ticks)

plt.yticks(new_ticks)#[-2,-1,1,2],['soBad','Bad','Good','veryGood']


ax = plt.gca()

ax.spines['top'].set_color('None')
ax.spines['right'].set_color('None')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(8)
    label.set_bbox(dict(facecolor='y',edgecolor='k',alpha = 0.5))

lb, = plt.plot(x, y2, label = 'bule')
lr, = plt.plot(x, y1, color='red', linestyle='--', label = 'rad')
plt.legend(handles=[lb,lr,],labels=['bule line','red line',],loc = 'best')

x3 = -0.413
y3 = x3**2
plt.scatter([x3,],[y3,],s=50, color='r')
plt.annotate('this is scater in line (  % s, % s  )'% (x3,y3),
xy=(x3,y3),xycoords='data',xytext=(-30,-30),textcoords='offset points',
fontsize = 8,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0.2'))

plt.text(0.5,-0.5,'this is plot text',fontdict={'size' : 8,'color' : 'b'})

plt.show()