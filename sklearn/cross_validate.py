from __future__ import print_function
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
print(knn_model.score(X_test, y_test))

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(knn_model, X, y, cv=5, scoring='accuracy')
print(scores.mean())
