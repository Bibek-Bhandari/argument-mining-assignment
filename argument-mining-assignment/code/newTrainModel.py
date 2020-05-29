from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

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

data_X = data_X[:25000]
data_Y = data_Y[:25000]
feature_extraction = TfidfVectorizer()
data_X_vector = feature_extraction.fit_transform(data_X)
print(data_X_vector)
joblib.dump(feature_extraction, 'feature_extraction.pkl')
classifier = SVC(probability=True, kernel='rbf')
#classifier=MLPClassifier(alpha=1, max_iter=1000)
classifier.fit(data_X_vector, data_Y)
joblib.dump(classifier, 'my_dumped_classifier.pkl')
