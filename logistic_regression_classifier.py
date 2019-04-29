from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

    # input(len(x_train))
    # input(len(x_train[0]))
    # input(len(x_train[0][0]))
    # input(type(x_train[0][0][0]))
    # input(len(y_train))


# ## loading testing dataset
with open("project635/case01/Dataset/2.csv", newline='') as testing_file:
    testing_data = list(csv.reader(testing_file))

    test_data = np.array_split(testing_data, [432], axis = 1)
    x_test = test_data[0].astype("float")
    y_test = test_data[1].astype("int")

#perform random under smapling

undersample = RandomUnderSampler(ratio={0:1915, 1:383})
x_train, y_train = undersample.fit_resample(x_train, y_train)


###logistic regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)

target_name = ['class 0', 'class 1']
print(classification_report(y_test, y_predict, target_names=target_name))

confusionMatrix = confusion_matrix(y_test, y_predict)
print(confusionMatrix)


true_positive, true_negative, false_positive, false_negative = confusionMatrix.ravel()


MCC = matthews_corrcoef(y_test, y_predict, sample_weight=None)
sensitivity = true_positive / (true_positive + false_negative)
specificity = true_negative / (false_positive + true_negative)

print("Validation Measures for Logistic Regression:")
print("Matthews Correlation Coefficient: ", MCC)
print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)
