##############################################################
# Appendix B Time Locality and Amount of Training Data       #
#                                                            #
# Analysis script to evaluate the time locality and          #
# amount of training issue reports. Note that for            #
# different time intervals DataLoader.py file should be      #
# changed before each evaluation.                            #
#                                                            #
# Before running, do not forget to change the DataLoader.py  #
# and TextPreProcessor.py files according to the specifics   # 
# of your data.                                              # 
##############################################################

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from time import gmtime, strftime, time
from DataLoader import *
from TextPreProcessor import TextPreProcessor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from mlxtend.classifier import StackingClassifier

inputfileName = "data/issues.csv"

print("Program starts:" + strftime("%Y-%m-%d %H:%M:%S", gmtime()))

dataLoader = DataLoader()
dataFilterer = DataFilterer()
preprocessor = TextPreProcessor()

# load the dataset
entireDataset = dataLoader.load(inputfileName)
print("Entire dataset length: " + str(len(entireDataset)))

# filter training issue records
trainDataset = dataFilterer.selectTrainingDatasetRecords(entireDataset)
trainDataset = dataFilterer.selectRecordsHavingAtLeastNValuesInColumn(trainDataset, CNAME_TEAMCODE)
# text preprocessing
trainDataset[CNAME_SUBJECT_DESCRIPTION] = trainDataset[CNAME_SUBJECT_DESCRIPTION].apply(preprocessor.filterNoise)

# print to check training records
print("Train dataset length : " + str(len(trainDataset)))
print("Train dataset loaded:" + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
print(trainDataset[CNAME_SUBJECT_DESCRIPTION].head(3))

# filter test issue records
testDataset = dataFilterer.selectTestDatasetRecords(entireDataset)
#text preprocessing
testDataset[CNAME_SUBJECT_DESCRIPTION] = testDataset[CNAME_SUBJECT_DESCRIPTION].apply(preprocessor.filterNoise)

# print to check test records
print("Test length: " + str(len(testDataset)))
print("Test dataset loaded:" + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
print(testDataset[CNAME_SUBJECT_DESCRIPTION].head(3))

# specify the input textual data to train (X_train) and related classes (Y_train)
X_train = trainDataset[CNAME_SUBJECT_DESCRIPTION].values
Y_train = trainDataset[CNAME_TEAMCODE].values

# Tf-idf conversion for training dataset
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_tfidf_train = vectorizer.fit_transform(X_train)
voc = vectorizer.vocabulary_

# specify the input textual data to test (X_test) and related classes (Y_test)
X_test = testDataset[CNAME_SUBJECT_DESCRIPTION].values
Y_test = testDataset[CNAME_TEAMCODE].values

# Td-idf conversion for test dataset
vectorizer = TfidfVectorizer(ngram_range=(1, 2), vocabulary=voc)
X_tfidf_test = vectorizer.fit_transform(X_test)

print("Ready to train and test:" + strftime("%Y-%m-%d %H:%M:%S", gmtime()))

###################################################
# Part 1 - Training                               #
###################################################
LinSvc = LinearSVC()

print('LinearSVC' + ":Training starts:" + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
start_time = time()
LinSvc.fit(X_tfidf_train, Y_train)
end_time = time()
print('LinearSVC' + ":Training ends:" + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
total_seconds = end_time - start_time
hours, rest = divmod(total_seconds, 3600)
minutes, seconds = divmod(rest, 60)
print("Training time: ", hours, minutes, seconds)

predictions = LinSvc.predict(X_tfidf_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

