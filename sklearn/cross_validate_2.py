from __future__ import print_function
from sklearn.learning_curve import  validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target

range_gamma = np.logspace(-6,-2.3,5)
train_loss,test_loss = validation_curve(
    SVC(), X, y, param_name='gamma', param_range=range_gamma,
    cv=10, scoring='mean_squared_error')

train_loss_mean = -np.mean(train_loss,axis=1)
test_loss_mean = -np.mean(test_loss,axis=1)

plt.plot(range_gamma,train_loss_mean,'x--',color='g',label='Train')
plt.plot(range_gamma,test_loss_mean,'x--',color='r',label='Test')

plt.xlabel('gamma')
plt.ylabel('loss')
plt.legend(loc="best")

plt.show()


