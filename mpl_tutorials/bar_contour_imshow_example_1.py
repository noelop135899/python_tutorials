import matplotlib.pyplot as plt
import numpy as np

n = 12
X = np.arange(n)
Y1 = (1 - X/float(n))*  np.random.uniform(0.5,1.0,n)
Y2 = (1 - X/float(n)) * np.random.uniform(0.5,1.0,n)

fig = plt.figure()

brplt = fig.add_subplot(211)
ctplt = fig.add_subplot(223)
imsplt = fig.add_subplot(224)

"""
fig.set_xticks(())
plt.yticks(())
ax = plt.gca()
ax.spines['top'].set_color('None')
ax.spines['right'].set_color('None')
ax.spines['left'].set_color('None')
ax.spines['bottom'].set_color('None')
"""


#plt.figure(num='bar')
#ticks_and_apines()
brplt.bar(X,+Y1,facecolor='r',edgecolor='w')
brplt.bar(X,-Y2,facecolor='b',edgecolor='w')

# ha: horizontal alignment
# va: vertical alignment
for x,y in zip(X,Y1):
    brplt.text(x+0.04,y+0.05,'%.2f'% y,ha='center',va='bottom')

for x,y in zip(X,Y2):
    brplt.text(x+0.04,-y-0.05,'%.2f'% y,ha='center',va='top')

#plt.figure(num='contour')
#ticks_and_apines()
m = 256
cx = np.linspace(-3,3,n)
cy = np.linspace(-3,3,n)
cX,cY = np.meshgrid(cx,cy)

def f(x,y):
    # the height function
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)

ctplt.contourf(cX,cY,f(cX,cY),18,alpha=0.75,cmap=plt.cm.winter)
c = ctplt.contour(cX,cY,f(cX,cY),16,colors='r',linewithd='.5')
ctplt.clabel(c,inline='True',fontsiza=10)

#plt.figure(num='imshow')
#ticks_and_apines()
i = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)
imsplt.imshow(i,interpolation='nearest',cmap='bone',origin='upper')


plt.show()
