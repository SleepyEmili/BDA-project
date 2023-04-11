import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

data = pd.read_csv('virusshare.csv', sep=',',skiprows=1, header=None).to_numpy()

X = data[:,1:]
Y = data[:,0]

kf = KFold(n_splits=5)
model = MLPClassifier(solver='sgd', learning_rate = "adaptive", learning_rate_init = 0.1, alpha=1e-5,hidden_layer_sizes=(10, 2), random_state=1,max_iter=500)
acc_scores = []
f1_scores = []

for train_index, test_index in kf.split(X):
    train_X, test_X = X[train_index], X[test_index]
    train_Y, test_Y = Y[train_index], Y[test_index]
    model.fit(train_X,train_Y)
    pred_values = model.predict(test_X)
    acc = accuracy_score(pred_values , test_Y)
    f1 = f1_score(pred_values , test_Y)
    acc_scores.append(acc)
    f1_scores.append(f1)

avg_acc_score = sum(acc_scores)/5
avg_f1_score = sum(f1_scores)/5

print('Method: MLP')
print('accuracy of each fold - {}'.format(acc_scores))
print('Avg accuracy : {}'.format(avg_acc_score))
print('f1 of each fold - {}'.format(f1_scores))
print('Avg f1 : {}'.format(avg_f1_score))