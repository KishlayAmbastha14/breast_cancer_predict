from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle
import pandas as pd 
import numpy as np

cancer = pd.read_csv('breast_cancer_dataframe.csv')

cancer_subset = cancer[['mean radius','mean perimeter','mean concave points', 'worst perimeter', 'worst area','target']]
cancer_subset

x = cancer_subset.drop(['target'],axis=1)  # here we split dataframe into x and y 
y = cancer_subset['target']                 # here y means we want to predict this 0 = no cancer (Benign tumors)
                              
# Splitting X_TRAIN, X_TEST, Y_tRAIN , Y_TEST

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)

# LOGISTSIC REGRESSION WITH NORMAL VALUES

lr_model = LogisticRegression(random_state = 51, penalty = 'l1',solver='liblinear', max_iter=400)
lr_model.fit(X_train,y_train)
y_pred_lr = lr_model.predict(X_test)
accuracy_score(y_test,y_pred_lr)

pickle.dump(lr_model,open('lr_model.pkl','wb'))