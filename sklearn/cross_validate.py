from __future__ import print_function
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score



iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
print(knn_model.score(X_test, y_test))


k_range = range(1,31)
k_scores = []

for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn_model, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

plt.plot(k_range,k_scores)
plt.show()

#