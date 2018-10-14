import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np

df_wine=pd.read_csv(r'C:\Users\WIN10\Downloads\wine.csv',sep=',')
X,y=df_wine.iloc[:,:-1].values,df_wine.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

feat_labels=df_wine.columns[:-1]
params_rf={'n_estimators':[1,2,5,10,50,100,500,1000,2500,5000,10000]}
forest=RandomForestClassifier()
grid=GridSearchCV(estimator=forest,
                 param_grid=params_rf,
                 scoring="accuracy",
                 cv=10)
grid.fit(X_train,y_train)
in_sample_accuracy=list(grid.cv_results_["mean_test_score"]*100)

sys_time=list(grid.cv_results_["mean_score_time"]*1000)

pd.set_option('precision',2)
result=pd.DataFrame([params_rf["n_estimators"],in_sample_accuracy,sys_time],index=['n_estimators','in_sample_accuracy(%)','sys_time(ms)'])
result

best_model=grid.best_estimator_
print(grid.best_params_)

print("best score: %3f"%best_model.score(X_test,y_test))

importances=best_model.feature_importances_
indices=np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1,30,
                           feat_labels[f],
                           importances[indices[f]]))
    
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
       importances[indices],
       color='lightblue',
       align='center')
plt.xticks(range(X_train.shape[1]),
          feat_labels,rotation=90)
plt.xlim([-1,X_train.shape[1]])
plt.tight_layout()
plt.show()

print("My name is Yuhang Mao")
print("My NetID is: yuhangm2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

