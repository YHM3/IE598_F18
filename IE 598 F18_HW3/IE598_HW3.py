# -*- coding: utf-8 -*-
# 确定新数据规模
import urllib.request #python3不用urllib2
import sys

#read data from uci data repository
target_url=("https://archive.ics.uci.edu/ml/machine-learning-"
            "databases/undocumented/connectionist-bench/sonar/sonar.all-data")

data=urllib.request.urlopen(target_url) 

#arrange data into list for labels and lists for attributes
xList=[]
labels=[]
for line in data:
    #split on comma
    row=line.strip().split(",".encode()) # python3格式要强制转码
    xList.append(row)
    
sys.stdout.write("Number of Rows of Data = " + str(len(xList)) + '\n')
sys.stdout.write("Number of Columns of Data = " + str(len(xList[1])))
#Number of Rows of Data = 208
#Number of Columns of Data = 61


#确定每个属性的特征
#arrange data into list for labels and list of lists for attributes
nrow=len(xList)
ncol=len(xList[1])

type=[0]*3
colCounts=[]

for col in range (ncol):
    for row in xList:
        try:
            a=float(row[col])
            if isinstance(a,float):
                type[0]+=1
        except ValueError:
            if len(row[col])>0:
                type[1]+=1
            else:
                type[2]+=1
    colCounts.append(type)
    type=[0]*3
    
sys.stdout.write("Col#"+'\t'+"Number"+'\t'+
                 "Strings"+'\t'+"Other\n")
iCol=0
for types in colCounts:
    sys.stdout.write(str(iCol)+'\t\t'+str(types[0])+
                     '\t\t'+str(types[2])+"\n")
    iCol+=1
# =============================================================================
# Col#    Number  Strings Other
# 0               208             0
# 1               208             0
# 2               208             0
# 3               208             0
# 4               208             0
# 5               208             0
# 6               208             0
# 7               208             0
# 8               208             0
# 9               208             0
# 10              208             0
# 11              208             0
# 12              208             0
# 13              208             0
# 14              208             0
# 15              208             0
# 16              208             0
# 17              208             0
# 18              208             0
# 19              208             0
# 20              208             0
# 21              208             0
# 22              208             0
# 23              208             0
# 24              208             0
# 25              208             0
# 26              208             0
# 27              208             0
# 28              208             0
# 29              208             0
# 30              208             0
# 31              208             0
# 32              208             0
# 33              208             0
# 34              208             0
# 35              208             0
# 36              208             0
# 37              208             0
# 38              208             0
# 39              208             0
# 40              208             0
# 41              208             0
# 42              208             0
# 43              208             0
# 44              208             0
# 45              208             0
# 46              208             0
# 47              208             0
# 48              208             0
# 49              208             0
# 50              208             0
# 51              208             0
# 52              208             0
# 53              208             0
# 54              208             0
# 55              208             0
# 56              208             0
# 57              208             0
# 58              208             0
# 59              208             0
# 60              0               0
# =============================================================================


#数值型和类别型属性的统计信息 
#generate summary statistics for column 3 (e.g.)
import numpy as np

col=3
colData=[]
for row in xList:
    colData.append(float(row[col]))
    
colArray=np.array(colData)
colMean=np.mean(colArray)
colsd=np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' +
                 "Standard Deviation = " + '\t' + str(colsd) + "\n")

#calculate quantile boundaries
ntiles=4

percentBdry=[]

for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
    
sys.stdout.write("Boundaries for 4 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write("\n")

#run again with 10 euqal intervals
ntiles=10
percentBdry=[]

for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
    
sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write("\n")


#The last column contains categorical variables

col=60
colData=[]
for row in xList:
    colData.append(row[col])
    
unique=set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)

#count up the number of elements having each value

catDict=dict(zip(list(unique),range(len(unique))))

catCount=[0]*2

for elt in colData:
    catCount[catDict[elt]]+=1
sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount)


#“岩石 vs. 水雷”数据集的第4列的分位数图
import pylab
import scipy.stats as stats
#generate summary statistics for column 3 (e.g.)
col=3
colData=[]
for row in xList:
    colData.append(float(row[col]))

stats.probplot(colData,dist="norm",plot=pylab)
pylab.show()


#用Python Pandas读入数据、分析数据
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot

#read rocks versus mines data into pandas data frame
rocksVMines=pd.read_csv(target_url,header=None,prefix="V")

#print head and tail of data frame
print(rocksVMines.head())
print(rocksVMines.tail())

#print summary of data frame
summary=rocksVMines.describe()
print(summary)
# =============================================================================
#  V0      V1      V2      V3 ...      V57     V58     V59  V60
# 0  0.0200  0.0371  0.0428  0.0207 ...   0.0084  0.0090  0.0032    R
# 1  0.0453  0.0523  0.0843  0.0689 ...   0.0049  0.0052  0.0044    R
# 2  0.0262  0.0582  0.1099  0.1083 ...   0.0164  0.0095  0.0078    R
# 3  0.0100  0.0171  0.0623  0.0205 ...   0.0044  0.0040  0.0117    R
# 4  0.0762  0.0666  0.0481  0.0394 ...   0.0048  0.0107  0.0094    R
# 
# [5 rows x 61 columns]
#          V0      V1      V2      V3 ...      V57     V58     V59  V60
# 203  0.0187  0.0346  0.0168  0.0177 ...   0.0115  0.0193  0.0157    M
# 204  0.0323  0.0101  0.0298  0.0564 ...   0.0032  0.0062  0.0067    M
# 205  0.0522  0.0437  0.0180  0.0292 ...   0.0138  0.0077  0.0031    M
# 206  0.0303  0.0353  0.0490  0.0608 ...   0.0079  0.0036  0.0048    M
# 207  0.0260  0.0363  0.0136  0.0272 ...   0.0036  0.0061  0.0115    M
# 
# [5 rows x 61 columns]
#                V0          V1     ...             V58         V59
# count  208.000000  208.000000     ...      208.000000  208.000000
# mean     0.029164    0.038437     ...        0.007941    0.006507
# std      0.022991    0.032960     ...        0.006181    0.005031
# min      0.001500    0.000600     ...        0.000100    0.000600
# 25%      0.013350    0.016450     ...        0.003675    0.003100
# 50%      0.022800    0.030800     ...        0.006400    0.005300
# 75%      0.035550    0.047950     ...        0.010325    0.008525
# max      0.137100    0.233900     ...        0.036400    0.043900
# 
# [8 rows x 60 columns]
# =============================================================================


#实数值属性可视化
for i in range(208):
    #assign color based on "M" or "R" labels
    if rocksVMines.iat[i,60]=="M":
        pcolor="red"
    else:
        pcolor="blue"
    
    #plot rows of data as if they were series data
    dataRow=rocksVMines.iloc[i,0:60]
    dataRow.plot(color=pcolor)

plot.xlabel("Attribute Index")
plot.ylabel("Attribute Values")
plot.show()


#属性对的交汇图
#calculate correlations between real-valued attributes
dataRow2=rocksVMines.iloc[1,0:60]
dataRow3=rocksVMines.iloc[2,0:60]

plot.scatter(dataRow2,dataRow3)

plot.xlabel("2nd Attribute")
plot.ylabel("3rd Attribute")
plot.show()

dataRow21=rocksVMines.iloc[20,0:60]

plot.scatter(dataRow2,dataRow21)

plot.xlabel("2nd Attribute")
plot.ylabel("21st Attribute")
plot.show()


#分类问题标签和实数值属性之间的相关性
from random import uniform
#change target to numeric values
target=[]
for i in range(208):
    #assign 0 or 1 target value based on "M" or "R" labels
    if rocksVMines.iat[i,60]=="M":
        target.append(1.0)
    else:
        target.append(0.0)
        
#plot 35th attribute
dataRow=rocksVMines.iloc[0:208,35]
plot.scatter(dataRow,target)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()

#
#To improve the visualization, this version dithers the points a little
# and makes them somewhat transparent
target=[]
for i in range(208):
    
    #assign 0 or 1 target value based on "M" or "R" labels
    #and add some dither
    
    if rocksVMines.iat[i,60]=="M":
        target.append(1.0+uniform(-0.1,0.1))
    else:
        target.append(0.0+uniform(-0.1,0.1))

#plot 35th attribute with semi-opaque points
dataRow=rocksVMines.iloc[0:208,35]
plot.scatter(dataRow,target,alpha=0.5,s=120)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()


# 对属性2和属性3、属性2和属性21分别计算各自的皮尔森相关系数
from math import sqrt
mean2=0.0;mean3=0.0;mean21=0.0
numElt=len(dataRow2)
for i in range(numElt):
    mean2+=dataRow2[i]/numElt
    mean3+=dataRow3[i]/numElt
    mean21+=dataRow21[i]/numElt
    
var2=0.0;var3=0.0;var21=0.0
for i in range(numElt):
    var2+=(dataRow2[i]-mean2)**2/numElt
    var3+=(dataRow3[i]-mean3)**2/numElt
    var21=(dataRow21[i]-mean21)**2/numElt
    
corr23=0.0;corr221=0.0
for i in range (numElt):
    corr23+=(dataRow2[i]-mean2)*\
    (dataRow3[i]-mean3)/(sqrt(var2*var3)*numElt)
    corr221+=(dataRow2[i]-mean2)*\
    (dataRow21[i]-mean21)/(sqrt(var2*var21)*numElt)

sys.stdout.write("Correlation between attribute 2 and 3 \n")
print(corr23)
sys.stdout.write("\n")
sys.stdout.write("Correlation between attribute 2 and 21 \n")
print(corr221)
# =============================================================================
# Correlation between attribute 2 and 3 
# 0.7709381211911223
# 
# Correlation between attribute 2 and 21 
# 3.0257938526755095
# =============================================================================


#属性相关系数可视化
#calculate correlations between real-valued attributes
corMat=DataFrame(rocksVMines.corr())

#visualize correlations using heatmap
plot.pcolor(corMat)
plot.show()

print("My name is Yuhang Mao")
print("My NetID is: yuhangm2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")