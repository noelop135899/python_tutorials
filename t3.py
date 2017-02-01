import numpy as np

dims = 3
length =25
lineData = np.empty((dims, length))
lineData[:, 0] = np.random.rand(dims)
for index in range(1, length):
    step = ((np.random.rand(dims) - 0.5) * 0.1)
    lineData[:, index] = lineData[:, index - 1] + step
print (lineData)
