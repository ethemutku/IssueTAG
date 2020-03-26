# Load libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from time import gmtime, strftime, time
from DataLoader import *
from TextPreProcessor import TextPreProcessor
from sklearn import model_selection
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

datesChecked = str(input("Did you check the dates to filter at DataLoader.py file? (Y/N): "))
if datesChecked != 'Y':
    exit(0)

inputfileName = "data/issues.csv"

# N-fold cross validation
N_FOLDS = 10

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

# Specify the algorithms for comparison
#
MultNB = MultinomialNB()
#
DT = DecisionTreeClassifier()
#
Knn = KNeighborsClassifier(n_neighbors=12, algorithm='brute', metric='cosine')
#
LR = LogisticRegression()
#
RF = RandomForestClassifier()
#
LinSvc = LinearSVC()
CLinSvc = CalibratedClassifierCV(LinSvc)

SclfBest_3 = StackingClassifier(classifiers=[CLinSvc, LR, Knn],
                                use_probas=True,
                                meta_classifier=LR)

SclfSelect_3 = StackingClassifier(classifiers=[CLinSvc, Knn, MultNB],
                                use_probas=True,
                                meta_classifier=LR)

SclfBest_5 = StackingClassifier(classifiers=[CLinSvc, LR, Knn, RF, DT],
                                use_probas=True,
                                meta_classifier=LR)

SclfSelect_5 = StackingClassifier(classifiers=[CLinSvc, LR, Knn, RF, MultNB],
                                use_probas=True,
                                meta_classifier=LR)

for clf, label in zip([MultNB, Knn, LR, LinSvc, CLinSvc, DT, RF, SclfBest_3, SclfSelect_3, SclfBest_5, SclfSelect_5],
                      ['Multinomial NB',
                       'KNN',
                       'Logistic Regression',
                       'Linear SVC',
                       'Linear SVC - Calibrated',
                       'Decision Tree',
                       'Random Forest',
                       'Best3',
                       'Selected3',
                       'Best5',
                       'Selected5']]):
    scores = model_selection.cross_val_score(clf, X_tfidf_train, Y_train, cv=N_FOLDS, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
    
    print(label + ":Training starts:" + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    start_time = time()
    clf.fit(X_tfidf_train, Y_train)
    end_time = time()
    print(label + ":Training ends:" + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    total_seconds = end_time - start_time
    hours, rest = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    print("Training time: ", hours, minutes, seconds)
    #
    predictions = clf.predict(X_tfidf_test)
    print(accuracy_score(Y_test, predictions))
    print(confusion_matrix(Y_test, predictions))
    print(classification_report(Y_test, predictions))
    #
