# -*- coding: utf-8 -*-
"""
##########################
Created on Wed Jan 11 2023
@author: Prateek
##########################
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('Data.csv', index_col=0)

X = data['TokenizedTwt'].copy()
y = data['polarity'].copy()
y[y>0] = 1
y[y<0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.5, 
                                                    random_state=123)

cv = CountVectorizer()
train_data = cv.fit_transform(X_train)
test_data = cv.transform(X_test)

model = 'RF'

if model == 'RF':
    rfc = RandomForestClassifier(n_estimators=50,criterion='entropy')
    rfc.fit(train_data,y_train)
    preds = rfc.predict(test_data)
    joblib.dump(rfc, 'Random_Forest.pkl')

elif model =='SVM':
    svc = SVC()
    svc.fit(train_data, y_train)
    preds = svc.predict(test_data)
    joblib.dump(svc, 'Support_Vector_Machine.pkl')
    
elif model == 'LR':
    lr = LogisticRegression(penalty='l2', max_iter=1000)
    lr.fit(train_data, y_train)
    preds = lr.predict(test_data)
    joblib.dump(lr, 'Logistic_Regression.pkl')

print(accuracy_score(y_test,preds))


