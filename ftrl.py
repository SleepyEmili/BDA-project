import numpy as np
from datatable import Frame
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def testFTRL(model,modelName):
    kf = KFold(n_splits=5)
    acc_scores = []
    f1_scores = []

    for train_index, test_index in kf.split(X):
        model.reset()
        train_X, test_X = Frame(X[train_index]), Frame(X[test_index])
        train_Y, test_Y = Frame(Y[train_index]), Frame(Y[test_index])
        model.fit(train_X,train_Y)
        pred_values = model.predict(test_X).to_numpy()
        pred_values = np.rint(pred_values)
        acc = accuracy_score(pred_values, Y[test_index])
        f1 = f1_score(pred_values , Y[test_index])
        acc_scores.append(acc)
        f1_scores.append(f1)

    avg_acc_score = sum(acc_scores)/5
    avg_f1_score = sum(f1_scores)/5

    print('Method: '+modelName)
    print('accuracy of each fold - {}'.format(acc_scores))
    print('Avg accuracy : {}'.format(avg_acc_score))
    print('f1 of each fold - {}'.format(f1_scores))
    print('Avg f1 : {}'.format(avg_f1_score))
from datatable.models import Ftrl

for st in ["0.7","0.75","0.8","0.85","0.9","0.95"]:
    data1 = pd.read_csv('virusshare.csv'.format(st), sep=',',skiprows=1, header=None).to_numpy()
    data2 = pd.read_csv('X_pca{}.csv'.format(st), sep=',',skiprows=1, header=None).to_numpy()
    X = data2
    Y = data1[:,0]
    print(st)
    testFTRL(Ftrl(),"FTRL")
