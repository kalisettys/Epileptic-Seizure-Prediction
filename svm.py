import numpy as np

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef


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

#perform random under smapling
##undersampling performed only on training data
undersample = NearMiss(ratio={0:1149, 1:383})
x_train, y_train = undersample.fit_resample(x_train, y_train)

##performing normalization
##on both test and train data

normalize_train_data = normalize(x_train)
normalize_test_data = normalize(x_test)

##standardizing data:
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


##SVM
svm = SVC(kernel="rbf", C = 1.0, gamma=0.1)
svm.fit(x_train, y_train)
y_predict = svm.predict(x_test)


target_name = ['class 0', 'class 1']
print(classification_report(y_test, y_predict, target_names=target_name))

confusionMatrix = confusion_matrix(y_test, y_predict)
print(confusionMatrix)


true_positive, true_negative, false_positive, false_negative = confusionMatrix.ravel()


MCC = matthews_corrcoef(y_test, y_predict, sample_weight=None)
sensitivity = true_positive / (true_positive + false_negative)
specificity = true_negative / (false_positive + true_negative)

print("Validation Measures for Support Vector Machines (SVM):")
print("Matthews Correlation Coefficient: ", MCC)
print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)




# ## 5-fold cross validation
# seed = 7
# np.random.seed(seed)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#


