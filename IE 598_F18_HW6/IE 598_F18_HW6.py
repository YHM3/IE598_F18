# -*- coding: utf-8 -*-
#load dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
iris_dataset=load_iris()

X=iris_dataset['data']
y=iris_dataset['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y,random_state=42)

from sklearn.tree import DecisionTreeClassifier
in_sample_accuracy=[]
out_of_sample_accuracy=[]
ind=list(range(1,11))
ind.append('mean')
ind.append('std')
for i in range(1,11):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y,random_state=i)
    dt=DecisionTreeClassifier(max_depth=4,random_state=0)
    dt.fit(X_train,y_train)
    in_sample_accuracy.append(dt.score(X_train,y_train))
    out_of_sample_accuracy.append(dt.score(X_test,y_test))
in_sample_accuracy.append(np.mean(in_sample_accuracy))
in_sample_accuracy.append(np.std(in_sample_accuracy[:-1]))
out_of_sample_accuracy.append(np.mean(out_of_sample_accuracy))
out_of_sample_accuracy.append(np.std(out_of_sample_accuracy[:-1]))
accuracy=pd.DataFrame([in_sample_accuracy,out_of_sample_accuracy,],
                        columns=ind,
                        index=['in_sample_accuracy','out_of_sample_accuracy'])
pd.set_option('precision',3)
accuracy

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y,random_state=42)
model=[]
for i in range(1,11):
    tree=DecisionTreeClassifier(max_depth=4,random_state=i)
    model.append(tree.fit(X_train,y_train))
in_sample_accuracy=[]
out_of_sample_accuracy=[]
for i in model:
    in_sample_accuracy.append(i.score(X_train,y_train))
    out_of_sample_accuracy.append(i.score(X_test,y_test))

ind=list(range(1,11))
ind.append('mean')
ind.append('std')
in_sample_accuracy.append(np.mean(in_sample_accuracy))
in_sample_accuracy.append(np.std(in_sample_accuracy[:-1]))
out_of_sample_accuracy.append(np.mean(out_of_sample_accuracy))
out_of_sample_accuracy.append(np.std(out_of_sample_accuracy[:-1]))
accuracy=pd.DataFrame([in_sample_accuracy,out_of_sample_accuracy,],
                        columns=ind,
                        index=['in_sample_accuracy','out_of_sample_accuracy'])
pd.set_option('precision',3)
accuracy

from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
dot_data=export_graphviz(tree,class_names=['setosa', 'versicolor', 'virginica'],
                feature_names=iris_dataset.feature_names,filled=True,out_file=None)
graph=graph_from_dot_data(dot_data)
graph.write_png('tree.png');

from sklearn.model_selection import cross_val_score
CVS=[]
scores=cross_val_score(DecisionTreeClassifier(max_depth=4),X_train,y_train,cv=10)
CVS.append(scores)
pd.set_option('precision',3)
result=pd.DataFrame(CVS,columns=['split1','split2','split3','split4','split5','split6','split7','split8','split9','split10'],)
result['mean']=result.mean(1)
result['std']=result.std(1)
## run the DecisionTree
dt=DecisionTreeClassifier(max_depth=4)
dt.fit(X_train,y_train)
result['Out-of-sample-accuracy']=dt.score(X_test,y_test)
result

param_grid={'criterion':['gini','entropy'],
            'max_depth':list(range(2,7)),
            'random_state':list(range(1,11))}
from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(DecisionTreeClassifier(),param_grid,cv=10)
grid_search.fit(X_train,y_train)
best=grid_search.best_params_
best['in-sample-accuracy']=grid_search.best_score_
## refit best
dt_best=DecisionTreeClassifier(max_depth=best['max_depth'],criterion=best['criterion'],random_state=best['random_state'])
dt_best.fit(X_train,y_train)
best['out-of-sample-accuracy']=dt_best.score(X_test,y_test)

pd.DataFrame([list(best.values())],columns=list(best.keys()))


print("My name is Yuhang Mao")
print("My NetID is: yuhangm2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
