import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

data = pd.read_csv('virusshare.csv', sep=',',skiprows=1, header=None).to_numpy()

X = data[:,1:]
Y = data[:,0]

kf = KFold(n_splits=5)
model = LogisticRegression(solver = "liblinear")

acc_score = []
for train_index, test_index in kf.split(X):
    train_X, test_X = X[train_index], X[test_index]
    train_Y, test_Y = Y[train_index], Y[test_index]
    model.fit(train_X,train_Y)
    pred_values = model.predict(test_X)
    acc = accuracy_score(pred_values , test_Y)
    acc_score.append(acc)

avg_acc_score = sum(acc_score)/5

print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))
