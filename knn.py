from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import matthews_corrcoef

import numpy as np
import csv



## loading training dataset
with open("project635/case01/Dataset/1.csv", newline='\n') as training_file:
    training_data = list(csv.reader(training_file))

    train_data = np.array_split(training_data, [432], axis=1)
    x_train = train_data[0].astype("float")
    y_train = train_data[1].astype("int")

# ## loading testing dataset
with open("project635/case01/Dataset/2.csv", newline='') as testing_file:
    testing_data = list(csv.reader(testing_file))

    test_data = np.array_split(testing_data, [432], axis = 1)
    x_test = test_data[0].astype("float")
    y_test = test_data[1].astype("int")

undersample = RandomUnderSampler(ratio={0:3830, 1:383})
x_train, y_train = undersample.fit_resample(x_train, y_train)


## K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
print("Fitting KNN")
knn.fit(x_train, y_train)
y_true_value = y_test
print("Predicting KNN")
y_predict = knn.predict(x_test)

#Classification Report
target_name = ['class 0', 'class 1']
print(classification_report(y_true_value, y_predict, target_names=target_name))

confusionMatrix = confusion_matrix(y_test, y_predict)
print(confusionMatrix)

true_positive, true_negative, false_positive, false_negative = confusionMatrix.ravel()



MCC = matthews_corrcoef(y_test, y_predict, sample_weight=None)
sensitivity = true_positive / (true_positive + false_negative)
specificity = true_negative / (false_positive + true_negative)

print("Validation Measures for k-Nearest Neighbors (kNN):")
print("Matthews Correlation Coefficient: ", MCC)
print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)
