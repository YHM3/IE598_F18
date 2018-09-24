import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv(r'C:\Users\WIN10\Downloads\concrete.csv',sep=',')
df.columns = ['cement','slag','ash','water','superplastic','coarseagg','fineagg','age','strength']

type(df)
print(df.keys())
df.shape
df.info()
df.head()

#EDA
data=df[['cement','slag', 'ash', 'water', 'superplastic', 'coarseagg',
      'fineagg', 'age']] 
target=df['strength']
X=data.values
y=target.values
#Visual EDA
_=pd.plotting.scatter_matrix(df,c=y,figsize=[10,10],s=15,marker='D')
sns.boxplot(df['strength'])
#plt.clf() #clears the plot area
df['cement'].plot.hist(bins=50)
plt.xlabel('Cement')
plt.show()
#数据处理，为null时一般用dropna，但是这里为零，假设为零需要消去
#是事实上可能并不需要，再咨询相关专家后，得知这里为零并非异常，因而在后续处理时仍使用全部数据
df_m=df[(df != 0).all(1)]
df_m

# heatmap
cols=['cement','slag', 'ash', 'water', 'superplastic', 'coarseagg',
      'fineagg', 'age']
cm=np.corrcoef(data.values.T)
sns.set(font_scale=1.5)
hm=sns.heatmap(cm,
               cbar=True,
               annot=True,
               square=True,
               fmt='.2f',
               annot_kws={'size':10},
               yticklabels=cols,
               xticklabels=cols)
plt.show()

#Train/test split
from sklearn.model_selection import train_test_split as tts
X_train,X_test,y_train,y_test=tts(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
slr=LinearRegression()

slr.fit(X,y) # all data
y_pred=slr.predict(X) # all data predict

np.set_printoptions(precision=3) #保留三位小数
print('Slope, whole set:',slr.coef_)
print('Intercept, whole set:%.3f'%slr.intercept_)

slr.fit(X_train,y_train) # training set
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

print('Slope, training set:',slr.coef_)
print('Intercept, training set:%.3f'%slr.intercept_)

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=0, xmax=100, color='black', lw=2)
plt.xlim([0, 100])
plt.show()

from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)))

from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)))

from sklearn.linear_model import Ridge
alpha=[0.01,0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]
score_train=[]
score_test=[]
coef=[]
intercept=[]
mse_train=[]
mse_test=[]
for i in alpha:
    ridge=Ridge(alpha=i,normalize=True).fit(X_train, y_train)#不归一化没效果
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    
    ## plot the residual errors
    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('alpha='+str(i))
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=0, xmax=100, color='black', lw=2)
    plt.xlim([0,100])
    plt.show()
    
    ## calculate mse, need normalize
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))
    score_train.append(ridge.score(X_train, y_train))
    score_test.append(ridge.score(X_test, y_test))
    coef.append(ridge.coef_)
    intercept.append(ridge.intercept_)
    
#制表输出结果，最优为0.01
Ridge_result=pd.DataFrame([score_train,score_test,mse_train,mse_test,coef,intercept],
                        columns=alpha,
                        index=['R^2_train','R^2_test','mse_train','mse_test',
                               'coef','intercept'])
    
Ridge_result.iloc[0:4]#输出train&test

for i in range(len(alpha)):
    print('When alpha=',alpha[i],'\nSlope:')
    print(Ridge_result.iloc[4,i])
    print('Intercept:\n%.3f'%slr.intercept_,'\n')

from sklearn.linear_model import Lasso
alpha=[0.01,0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]
score_train=[]
score_test=[]
coef=[]
intercept=[]
mse_train=[]
mse_test=[]
for i in alpha:
    lasso=Lasso(alpha=i,normalize=True).fit(X_train, y_train)#不归一化没效果
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    
    ## plot the residual errors
    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('alpha='+str(i))
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=0, xmax=80, color='black', lw=2)
    plt.xlim([0,80])
    plt.show()
    
    ## calculate mse, need normalize
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))
    score_train.append(lasso.score(X_train, y_train))
    score_test.append(lasso.score(X_test, y_test))
    coef.append(lasso.coef_)
    intercept.append(lasso.intercept_)
    
Lasso_result=pd.DataFrame([score_train,score_test,mse_train,mse_test,coef,intercept],
                        columns=alpha,
                        index=['R^2_train','R^2_test','mse_train','mse_test',
                               'coef','intercept'])

Lasso_result.iloc[0:4]#输出train&test

for i in range(len(alpha)):
    print('When alpha=',alpha[i],'\nSlope:')
    print(Lasso_result.iloc[4,i])
    print('Intercept:\n%.3f'%slr.intercept_,'\n')

from sklearn.linear_model import ElasticNet
l1_ratio=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
score_train=[]
score_test=[]
coef=[]
intercept=[]
mse_train=[]
mse_test=[]
for i in alpha:
    elanet=ElasticNet(alpha=1.0, l1_ratio=i).fit(X_train, y_train)#不归一化没效果
    y_train_pred = elanet.predict(X_train)
    y_test_pred = elanet.predict(X_test)
    
    ## plot the residual errors
    plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('alpha='+str(i))
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=0, xmax=100, color='black', lw=2)
    plt.xlim([0,100])
    plt.show()
    
    ## calculate mse, need normalize
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))
    score_train.append(elanet.score(X_train, y_train))
    score_test.append(elanet.score(X_test, y_test))
    coef.append(elanet.coef_)
    intercept.append(elanet.intercept_)
    
Elanet_result=pd.DataFrame([score_train,score_test,mse_train,mse_test,coef,intercept],
                        columns=alpha,
                        index=['R^2_train','R^2_test','mse_train','mse_test',
                               'coef','intercept'])

Elanet_result.iloc[0:4]#输出train&test

for i in range(len(alpha)):
    print('When alpha=',alpha[i],'\nSlope:')
    print(Elanet_result.iloc[4,i])
    print('Intercept:\n%.3f'%slr.intercept_,'\n')

