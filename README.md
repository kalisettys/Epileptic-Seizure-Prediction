Data:
------

1. The EEG data is to be downloaded from www.physionet.org/pn6/chbmit/.
It is about 50GB in size so start downloading them soon.
The data were used in “Application of Machine Learning to Epileptic Seizure Detection” paper by Shoeb and Guttag, which you need to study (same link) to understand both the domain and the data.


REQUIREMENTS:
-------------

1. Preprocess the data, as in the paper
2. Design 3 different classification models, including SVM model (as in the paper), to recognize/predict
seizure during the first 24 hours of EEG recordings (meaning that you will use it for both training and validation) for the following 4 patients:
3. Report your training/validating results, plus prediction results for each patient (in ppt).
4. You can use any programming language/software (Python/WEKA/MATLAB).

------------------------------------------------------------------------------------------------------------------------------


Shilpa Kalisetty 
CMSC 635: Data Mining Project: Epileptic Seizure Detection in Python
Dr. Cios: May 2nd, 2019


EXECUTING THE PROJECT:
----------------------

1. Download the project ‘zip file’

2. Unzip the folder

3. Open the project in your Python IDE (I used PyCharm for MAC OS)

4. Travel to the data_preprocessing.py to execute the program to generate all the .csv files :

	NOTE: I have manually separated all the 42 .csv files into training and testing as I was getting weird errors:

	-  Training Data = "1.csv" --> consists of files from chb01_01.csv to chb01_24.csv
	- Testing Data = "2.csv" --> consists of files from chb01_25.csv to chb01_46.csv

5. Now to test the classifiers: you can do two things:

a. GO to the "classifiers.py" file and run the file and you can see results for ALL the classifiers at ONCE

b. GO to each of the named classifiers "svm.py", "decision_tree.py", "knn.py", or "logistic_regression_classifer.py" to execute the program


	
==> Second approach is preferred so you can clearly see the differentiation between each classifier and their results.

