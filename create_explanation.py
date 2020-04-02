##############################################################
# Section 5 Explaining Team Assignments                      #
#                                                            #
# Script to create explanations for input issue records      #
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
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

 
summary = str(input("Please enter the summary of the issue report: "))
description = str(input("Please enter the description of the issue report: "))

inputfileName = "data/issues.csv"

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
print(trainDataset[CNAME_SUBJECT_DESCRIPTION].head(3))

# specify the input textual data to train (X_train) and related classes (Y_train)
X_train = trainDataset[CNAME_SUBJECT_DESCRIPTION].values
Y_train = trainDataset[CNAME_TEAMCODE].values

# Tf-idf conversion for training dataset
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(X_train)
voc = vectorizer.vocabulary_

# Use calibrated Linear SVC to obtain the probabilities for predictions
LinSvc = LinearSVC() # Linear SVC model 
CLinSvc = CalibratedClassifierCV(LinSvc)
# fit the model
cSvm.fit(train_vectors, Y_train)

# Preprocess input textual data
input_text = summary + description
X_test = [input_text]
X_test[0] = preprocessor.filterNoise(X_test[0])

# Td-idf conversion for test dataset
vectorizer = TfidfVectorizer(ngram_range=(1, 2), vocabulary=voc)
test_vectors = vectorizer.transform(X_test)

# Predict the team
pred = cSvm.predict(test_vectors)

# Explain the instance 
c = make_pipeline(vectorizer, cSvm)
class_names = c.classes_
explainer = LimeTextExplainer(class_names=class_names)
top_n_labels = 3 # Top n recommendations to explain
exp = explainer.explain_instance(X_test[0], c.predict_proba, num_features=6, top_labels=top_n_labels)

# starting from the best prediction to nth, explain the prediction
for i in range(0,top_n_labels):
    exp.as_pyplot_figure(label=exp.available_labels()[i])
