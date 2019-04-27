import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix



##splitting data into training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

## feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


## confusion matrix

#confusionMatrix = confusion_matrix(y_test, y_pred=)
#print(confusionMatrix)


## 5-fold cross validation
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

##SVM
svm = SVC(kernel="linear", c = 0.025, random_state=101)
#svm.fit(x_train, y_train)
#y_pred = svm.predict(x_test)



## K-Nearest Neighbors

knn = KNeighborsClassifier(n_neighbors=15)
#knn.fit(x_train, y_train)
#y_pred = knn.predict(x_test)



## Decision Trees

dtree = DecisionTreeClassifier(max_depth=10, random_state=101, max_features=None, min_samples_leaf=15)

#dtree.fit(x_train, y_train)
#y_pred = dtree.predict(x_test)