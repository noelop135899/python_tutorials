from __future__ import print_function
from sklearn import svm
from sklearn import datasets
import pickle
from sklearn.externals import joblib

iris = datasets.load_iris()
X = iris.data
y = iris.target

model = svm.SVC()
model.fit(X,y)

#method 1:pickle
#save
#with open('save_dir/iris_pickle.pickle','wb') as file:
 #   pickle.dump(model,file)
#read
with open('save_dir/iris_pickle.pickle','rb') as file:
    model_2 = pickle.load(file)
    print(X[0:1])
file.close()

#method 2 joblib:#faster
#save
#joblib.dump(model,'save_dir/iris_joblib.pkl')
#read
model_3 = joblib.load('save_dir/iris_joblib.pkl')
print(X[1:2])
