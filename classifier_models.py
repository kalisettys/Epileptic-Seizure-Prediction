import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


import csv


## loading training dataset
with open("project635/case01/Training.csv", newline='\n') as training_file:
    training_data = list(csv.reader(training_file))

    x_train = np.array_split(training_data, [432], axis=1)
    y_train = x_train[1]
    #input(y_train[10179:10185])


# ## loading testing dataset
with open("project635/case01/Testing.csv", newline='') as testing_file:
    testing_data = list(csv.reader(testing_file))
    x_test = np.array_split(testing_data, [432], axis = 1)
    y_test = x_test[1]



#
#
#
# ## confusion matrix
#
# #confusionMatrix = confusion_matrix(y_test, y_pred=)
# #print(confusionMatrix)
#
#
# ## 5-fold cross validation
# seed = 7
# np.random.seed(seed)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#
# ##SVM
# svm = SVC(kernel="rbf", c = 1.0, random_state=101)
# #svm.fit(x_train, y_train)
# #y_pred = svm.predict(x_test)
#
#
#
## K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_true_value = [x_test, y_test]
y_predict = knn.predict(x_test)

#Classification Report
target_name = ['class 0', 'class 1']
print(classification_report(y_true_value, y_predict, target_name))






#
# ## Decision Trees
#
# dtree = DecisionTreeClassifier(max_depth=10, random_state=101, max_features=None, min_samples_leaf=15)
#
# #dtree.fit(x_train, y_train)
#y_pred = dtree.predict(x_test)
