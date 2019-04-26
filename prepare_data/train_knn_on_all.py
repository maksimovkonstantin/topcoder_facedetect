import pickle
from sklearn import neighbors
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_path = '/wdata/mxnet_train_features.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)
X = data['X']
y = data['y']
y = [0 if el == -1 else 1 for i, el in enumerate(y)]
X_train = X
X_test = X
y_train = y
y_test = y

n_neighbors = 1
knn_algo = 'ball_tree'
print(len(X))

knn_clf = neighbors.KNeighborsClassifier(
                                 n_neighbors=n_neighbors,
                                 algorithm=knn_algo,
                                 weights='distance',
                                 n_jobs=-1)
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict(X_test)
print(n_neighbors, accuracy_score(y_test, y_pred))
with open('/wdata/model_original_{}.pkl'.format(n_neighbors), 'wb') as f:
    pickle.dump(knn_clf, f)
