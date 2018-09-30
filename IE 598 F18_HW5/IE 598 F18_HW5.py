# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
wine=pd.read_csv(r'C:\Users\WIN10\Downloads\wine.csv',sep=',')

wine.info()
wine.head()
X,y=wine.iloc[:,:-1],wine.iloc[:,-1]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_std=sc.fit_transform(X.values)
_=pd.plotting.scatter_matrix(wine,c=y.values,figsize=[20,20],s=15,marker='D')

_=sns.swarmplot(x='Class',y='Alcohol',data=wine)
_=plt.xlabel('Class')
_=plt.ylabel('alcohol')
plt.show()

_=sns.boxplot(x='Class',y='OD280/OD315 of diluted wines',data=wine)
_=plt.xlabel('Classes')
_=plt.ylabel('OD280/OD315 of diluted wines')
plt.show()

pd.set_option('precision',3)
X.describe()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X.values,y.values,test_size=0.2,stratify=y,random_state=42)
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
lr=LogisticRegression()
svm=LinearSVC()
lr.fit(X_train_std,y_train);
svm.fit(X_train_std,y_train);
# ouput accuracy
y_train_pred=lr.predict(X_train_std)
y_test_pred=lr.predict(X_test_std)
print("The accuracy score of logistic regression on TrainingSet: %.3f"
      %accuracy_score(y_train,y_train_pred),"\n\t\t\t\t\t  on TestSet: %.3f"
      %accuracy_score(y_test,y_test_pred))

y_train_pred=svm.predict(X_train_std)
y_test_pred=svm.predict(X_test_std)
print("The accuracy score of SVM on TrainingSet:%.3f"
      %accuracy_score(y_train,y_train_pred),"\n\t\t\t  on TestSet: %.3f"
      %accuracy_score(y_test,y_test_pred))

from sklearn.model_selection import cross_val_score as cvs
cvscores_5 = cvs(lr,X,y,cv=5)
print("The accuracy score of logistic regression on TrainingSet:%.3f"
      %np.mean(cvscores_5))
cvscores_5 = cvs(svm,X,y,cv=5)
print("The accuracy score of SVM on TrainingSet:%.3f"
      %np.mean(cvscores_5))

from sklearn.decomposition import PCA
pca=PCA()
pca.fit(X_std)
features=range(pca.n_components_)
plt.bar(features,pca.explained_variance_ratio_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()

transformed=pca.transform(X_std)
feature1=transformed[:,0]
feature2=transformed[:,1]
plt.scatter(feature1,feature2,c=y)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

pca=PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
svm.fit(X_train_pca,y_train);
# ouput accuracy
y_train_pred=lr.predict(X_train_pca)
y_test_pred=lr.predict(X_test_pca)
print("The accuracy score of logistic regression on TrainingSet: %.3f"
      %accuracy_score(y_train,y_train_pred),"\n\t\t\t\t\t  on TestSet: %.3f"
      %accuracy_score(y_test,y_test_pred))

y_train_pred=svm.predict(X_train_pca)
y_test_pred=svm.predict(X_test_pca)
print("The accuracy score of SVM on TrainingSet:%.3f"
      %accuracy_score(y_train,y_train_pred),"\n\t\t\t  on TestSet: %.3f"
      %accuracy_score(y_test,y_test_pred))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
transformed=lda.transform(X_std)
feature1=transformed[:,0]
feature2=transformed[:,1]
plt.scatter(feature1,feature2,c=y)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.show()

X_train_lda = lda.fit_transform(X_train_std,y_train)
X_test_lda = lda.transform(X_test_std)
lr.fit(X_train_lda,y_train);
svm.fit(X_train_lda,y_train);
# ouput accuracy
y_train_pred=lr.predict(X_train_lda)
y_test_pred=lr.predict(X_test_lda)
print("The accuracy score of logistic regression on TrainingSet: %.3f"
      %accuracy_score(y_train,y_train_pred),"\n\t\t\t\t\t  on TestSet: %.3f"
      %accuracy_score(y_test,y_test_pred))

y_train_pred=svm.predict(X_train_lda)
y_test_pred=svm.predict(X_test_lda)
print("The accuracy score of SVM on TrainingSet:%.3f"
      %accuracy_score(y_train,y_train_pred),"\n\t\t\t  on TestSet: %.3f"
      %accuracy_score(y_test,y_test_pred))

from sklearn.decomposition import KernelPCA
GM=[0.1,0.4,0.45,0.5,0.55,0.6,1,5,15]
lrTrain=[]
lrTest=[]
svmTrain=[]
svmTest=[]
for i in GM:
    scikit_kpca = KernelPCA(n_components=2, 
                  kernel='rbf', gamma=i)
    transformed = scikit_kpca.fit_transform(X_std)
    feature1=transformed[:,0]
    feature2=transformed[:,1]
    #plt.subplot(3,3,j)
    plt.figure(figsize=(2,2))
    plt.scatter(feature1,feature2,c=y)
    plt.title("wine via KernelPCA when gamma="+str(i))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()
    X_train_kpca = scikit_kpca.fit_transform(X_train_std,y_train)
    X_test_kpca = scikit_kpca.transform(X_test_std)
    lr.fit(X_train_kpca,y_train);
    svm.fit(X_train_kpca,y_train);
    # ouput accuracy
    y_train_pred=lr.predict(X_train_kpca)
    y_test_pred=lr.predict(X_test_kpca)
    lrTrain.append(accuracy_score(y_train,y_train_pred))
    lrTest.append(accuracy_score(y_test,y_test_pred))
    
    y_train_pred=svm.predict(X_train_kpca)
    y_test_pred=svm.predict(X_test_kpca)
    svmTrain.append(accuracy_score(y_train,y_train_pred))
    svmTest.append(accuracy_score(y_test,y_test_pred))
    
#save the result
result=pd.DataFrame([lrTrain,lrTest,svmTrain,svmTest],
                        columns=GM,
                        index=['lrTrain','lrTest','svmTrain','svmTest'])
result

lrTrain=[]
lrTest=[]
svmTrain=[]
svmTest=[]
for i in GM:
    scikit_kpca = KernelPCA(kernel='rbf', gamma=i)
    X_train_kpca = scikit_kpca.fit_transform(X_train_std,y_train)
    X_test_kpca = scikit_kpca.transform(X_test_std)
    lr.fit(X_train_kpca,y_train);
    svm.fit(X_train_kpca,y_train);
    # ouput accuracy
    y_train_pred=lr.predict(X_train_kpca)
    y_test_pred=lr.predict(X_test_kpca)
    lrTrain.append(accuracy_score(y_train,y_train_pred))
    lrTest.append(accuracy_score(y_test,y_test_pred))
    

    y_train_pred=svm.predict(X_train_kpca)
    y_test_pred=svm.predict(X_test_kpca)
    svmTrain.append(accuracy_score(y_train,y_train_pred))
    svmTest.append(accuracy_score(y_test,y_test_pred))
    
#save the result
result=pd.DataFrame([lrTrain,lrTest,svmTrain,svmTest],
                        columns=GM,
                        index=['lrTrain','lrTest','svmTrain','svmTest'])
result

print("My name is Yuhang Mao")
print("My NetID is: yuhangm2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")