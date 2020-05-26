from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

data_X = []
data_Y = []
with open('data/traindata.txt') as fp:
    # 1. iterate over file line-by-line
    # 2. strip line of newline symbols
    # 3. split line by spaces into list (of number strings)
    # 4. convert number substrings to int values
    # 5. convert map object to list

    for line in fp:
    	if(len(line.strip().split('\t')) > 1):
    		data_X.append(line.strip().split('\t')[0])
    		data_Y.append(line.strip().split('\t')[1])


feature_extraction = TfidfVectorizer()
data_X_vector = feature_extraction.fit_transform(data_X)



clf = SVC(probability=True, kernel='rbf')
clf.fit(data_X_vector, data_Y)


test_X = []
test_Y = []

with open('data/testdata.txt') as fp:
    # 1. iterate over file line-by-line
    # 2. strip line of newline symbols
    # 3. split line by spaces into list (of number strings)
    # 4. convert number substrings to int values
    # 5. convert map object to list

    for line in fp:
       	if(len(line.strip().split('\t')) > 1):
       		test_X.append(line.strip().split('\t')[0])
       		test_Y.append(line.strip().split('\t')[1])

test_X_vector = feature_extraction.transform(test_X)



predictions = clf.predict_proba(test_X_vector)
print('ROC-AUC yields ' + str(roc_auc_score(test_Y, predictions,multi_class="ovo")))