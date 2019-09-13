#!/usr/bin/env python
# coding: utf-8

# ## 1. Time Series Classification

# ### Importing Libraries

import sys
import glob
import natsort
import sklearn
import datetime
import warnings
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stat
import numpy.random as npr
import matplotlib.pyplot as plt
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import bootstrapped.compare_functions as bs_compare
from scipy.stats import norm
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve


# To suppress warnings
warnings.filterwarnings('ignore')


# ## (a) Download AReM data from GIT local repository

get_ipython().system(' git clone https://github.com/devikasathaye/Time-Series-Classification-AReM')


# ## (b) Data Management. Train and test split.

folders=[]

bending1='Time-Series-Classification-AReM/AReM/bending1'
bending2='Time-Series-Classification-AReM/AReM/bending2'
cycling='Time-Series-Classification-AReM/AReM/cycling'
lying='Time-Series-Classification-AReM/AReM/lying'
sitting='Time-Series-Classification-AReM/AReM/sitting'
standing='Time-Series-Classification-AReM/AReM/standing'
walking='Time-Series-Classification-AReM/AReM/walking'

folders.append(bending1)
folders.append(bending2)
folders.append(cycling)
folders.append(lying)
folders.append(sitting)
folders.append(standing)
folders.append(walking)

print(folders)


# Data management

df_train_list=[] # Train dataframes list
df_test_list=[] # Test dataframes list
df_list=[] # Entire dataset
train_target=[] # Target of train dataset- binary classification
test_target=[] # Target of test dataset- binary classification
train_all_target=[] # Target of train dataset- multi-class classification
test_all_target=[] # Target of test dataset- multi-class classification

# Data cleaning- extra comma at the end of a csv file is ignored/removed by using usecols=range(7)

# bending1 1
# bending2 2
# cycling 3
# lying 4
# sitting 5
# standing 6
# walking 7

# For binary classification, bending=1, other=0 #later

for folder in folders:
    for file in natsort.natsorted(glob.glob(folder+'\*.csv'),reverse=False):
        if(file==folder+'\dataset1.csv' or file==folder+'\dataset2.csv'):
            df_test_list.append(pd.read_csv(file, sep=',| ', usecols=range(7), header=None, skiprows=5, names=['time','avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23'], engine='python'))
            if(folder==bending1 or folder==bending2):
                if(folder==bending1):
                    test_all_target.append("bending1")
                else:
                    test_all_target.append("bending2")
                test_target.append("bending") # bending1 and bending2 have label 1, all others have label 0, for binary classification
            else:
                if(folder==cycling):
                    test_all_target.append("cycling")
                elif(folder==lying):
                    test_all_target.append("lying")
                if(folder==sitting):
                    test_all_target.append("sitting")
                if(folder==standing):
                    test_all_target.append("standing")
                if(folder==walking):
                    test_all_target.append("walking")
                test_target.append("other")
        elif(file==folder+'\dataset3.csv'):
            if(folder==bending1 or folder==bending2):
                df_train_list.append(pd.read_csv(file, sep=',| ', usecols=range(7), header=None, skiprows=5, names=['time','avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23'], engine='python'))
                if(folder==bending1):
                    train_all_target.append("bending1")
                else:
                    train_all_target.append("bending2")
                train_target.append("bending")
            else:
                df_test_list.append(pd.read_csv(file, sep=',| ', usecols=range(7), header=None, skiprows=5, names=['time','avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23'], engine='python'))
                if(folder==cycling):
                    test_all_target.append("cycling")
                elif(folder==lying):
                    test_all_target.append("lying")
                if(folder==sitting):
                    test_all_target.append("sitting")
                if(folder==standing):
                    test_all_target.append("standing")
                if(folder==walking):
                    test_all_target.append("walking")
                test_target.append("other")
        else:
            if(folder==bending1 or folder==bending2):
                if(folder==bending1):
                    train_all_target.append("bending1")
                else:
                    train_all_target.append("bending2")
                train_target.append("bending")
            else:
                if(folder==cycling):
                    train_all_target.append("cycling")
                elif(folder==lying):
                    train_all_target.append("lying")
                if(folder==sitting):
                    train_all_target.append("sitting")
                if(folder==standing):
                    train_all_target.append("standing")
                if(folder==walking):
                    train_all_target.append("walking")
                train_target.append("other")
            df_train_list.append(pd.read_csv(file, sep=',| ', usecols=range(7), header=None, skiprows=5, names=['time','avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23'], engine='python'))
        df_list.append(pd.read_csv(file, sep=',| ', usecols=range(7), header=None, skiprows=5, names=['time','avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23'], engine='python'))

#To show the correct split of train and test data
print("Train data-")
for q in range(len(df_train_list)):
    print(df_train_list[q].head(2))
    print("")
print("Size of train dataset",len(df_train_list))

print("")
print("=======================================================")

print("Test data-")
for r in range(len(df_test_list)):
    print(df_test_list[r].head(2))
    print("")
print("Size of test dataset",len(df_test_list))

print("")
print("=======================================================")

print("Train targets for binary classification")
print(train_target)
print("Size of train labels for binary classification",len(train_target))

print("")
print("=======================================================")

print("Test targets for binary classification")
print(test_target)
print("Size of test labels for binary classification",len(test_target))

print("")
print("=======================================================")

print("Train targets for multiclass classification")
print(train_all_target)
print("Size of train labels for multiclass classification",len(train_all_target))

print("")
print("=======================================================")

print("Test target for multiclass classification")
print(test_all_target)
print("Size of test labels for multiclass classification",len(test_all_target))

print("")
print("=======================================================")

print("Entire data-")
for s in range(len(df_test_list)):
    print(df_list[s].head(2))
    print("")
print("Size of entire dataset",len(df_list))


# ## (c) Feature Extraction

# ### ii. Extract the time-domain features minimum, maximum, mean, median, standard deviation, first quartile, and third quartile for all of the 6 time series in each instance.

columns=df_list[0].columns
print("All columns\n",columns)
print("")
cols=columns[1:]
print("All columns excluding time column\n",cols)

#Function to calculate minimum, returns list of list
def calc_min(dataframe_list):
    minimum=[]
    for j in dataframe_list[0].drop('time', axis=1, errors='ignore').columns:
        mini=[]
        for i in dataframe_list:
            mini.append(min(i[j]))
        minimum.append(mini)
    #print(minimum)
    return minimum


#Function to calculate maximum, returns list of list
def calc_max(dataframe_list):
    maximum=[]
    for j in dataframe_list[0].drop('time', axis=1, errors='ignore').columns:
        maxi=[]
        for i in dataframe_list:
            maxi.append(max(i[j]))
        maximum.append(maxi)
    #print(maximum)
    return maximum


#Function to calculate mean, returns list of list
def calc_mean(dataframe_list):
    mean=[]
    for j in dataframe_list[0].drop('time', axis=1, errors='ignore').columns:
        m=[]
        for i in dataframe_list:
            m.append(round(i[j].mean(),3))
        mean.append(m)
    #print(mean)
    return mean


#Function to calculate median, returns list of list
def calc_median(dataframe_list):
    median=[]
    for j in dataframe_list[0].drop('time', axis=1, errors='ignore').columns:
        me=[]
        for i in dataframe_list:
            me.append(round(i[j].median(),3))
        median.append(me)
    #print(median)
    return median

#Function to calculate Standard Deviation, returns list of list
def calc_std(dataframe_list):
    std=[]
    for j in dataframe_list[0].drop('time', axis=1, errors='ignore').columns:
        s=[]
        for i in dataframe_list:
            s.append(round(i[j].std(),3))
        std.append(s)
    #print(std)
    return std

#Function to calculate First Quartile, returns list of list
def calc_fq(dataframe_list):
    fq=[]
    for j in dataframe_list[0].drop('time', axis=1, errors='ignore').columns:
        f=[]
        for i in dataframe_list:
            #f.append(round(i[j].describe()[4],3)) # Another method to calculate first quartile
            f.append(round(i[j].quantile(q=0.25),3))
        fq.append(f)
    #print(fq)
    return fq

#Function to calculate Third Quartile, returns list of list
def calc_tq(dataframe_list):
    tq=[]
    for j in dataframe_list[0].drop('time', axis=1, errors='ignore').columns:
        t=[]
        for i in dataframe_list:
            #t.append(round(i[j].describe()[6],3)) # Another method to calculate third quartile
            t.append(round(i[j].quantile(q=0.75),3))
        tq.append(t)
    #print(tq)
    return tq

#Function to print the statistics table

def print_stats_table(args1,args2):
    stats=[]
    for o in range(len(args2[0])):
        for n in args2:
            stats.append(n[o])
    stats=list(map(list,zip(*stats)))
    statsDF=pd.DataFrame(stats, columns=args1)
    #print(statsDF)
    return statsDF

#Print stats table for raw features

args1=[]
args2=[]
for i in range(1,7,1):
    args1.append('Min'+str(i))
    args1.append('Max'+str(i))
    args1.append('Mean'+str(i))
    args1.append('Median'+str(i))
    args1.append('StdDev'+str(i))
    args1.append('FirstQuartile'+str(i))
    args1.append('ThirdQuartile'+str(i))

minlist=[]
maxlist=[]
meanlist=[]
medianlist=[]
stdlist=[]
fqlist=[]
tqlist=[]

minlist=calc_min(df_list)
maxlist=calc_max(df_list)
meanlist=calc_mean(df_list)
medianlist=calc_median(df_list)
stdlist=calc_std(df_list)
fqlist=calc_fq(df_list)
tqlist=calc_tq(df_list)

args2.append(minlist)
args2.append(maxlist)
args2.append(meanlist)
args2.append(medianlist)
args2.append(stdlist)
args2.append(fqlist)
args2.append(tqlist)

statsDF_raw=print_stats_table(args1,args2)
statsDF_raw

# Function to normalize features using MinMaxScaler
def norm_ft(df_fit, df_transform):
    minMaxSc = MinMaxScaler()
    minMaxSc.fit(df_fit)
    normDF=pd.DataFrame(minMaxSc.transform(df_transform), columns=df_transform.columns)
    return normDF

# Print normalized features stats table
statsDF_norm=norm_ft(statsDF_raw,statsDF_raw)
statsDF_norm


# ### iii. Estimate the standard deviation of each of the time-domain features you extracted from the data.

# Standard deviation of raw features
stdev_raw=round(statsDF_raw.std(),3)
print("Standard deviations of raw features")
print("Feature\t\tStdDev")
print(stdev_raw)

# Standard deviation of normalized features

stdev_norm=round(statsDF_norm.std(),3)
print("Standard deviations of normalized features")
print("Feature\t\tStdDev")
print(stdev_norm)


# ### Use Python's bootstrapped or any other method to build a 90% bootsrap confidence interval for the standard deviation of each feature

stdev_results=[]
std_boot_df=[] # List to display the results
std_boot_col=[]
std_boot_actual_std=[]
std_boot_est_std=[]
std_boot_lb=[]
std_boot_ub=[]
for col in statsDF_raw.columns:
    samples = statsDF_raw[col]
    bsr=bs.bootstrap(np.array(samples), stat_func=bs_stats.std, alpha=0.1)
    stdev_results.append(bsr)
    std_boot_col.append(col)
    std_boot_actual_std.append(stdev_raw[col])
    std_boot_est_std.append(bsr.value)
    std_boot_lb.append(bsr.lower_bound)
    std_boot_ub.append(bsr.upper_bound)
std_boot_df.append(std_boot_col) # List of feature names
std_boot_df.append(std_boot_actual_std) # List of actual standard deviation values
std_boot_df.append(std_boot_est_std) # List of estimated standard deviation values
std_boot_df.append(std_boot_lb) # List of lower bound of confidence interval
std_boot_df.append(std_boot_ub) # List of upper bound of confidence interval
std_boot_df=list(map(list,zip(*std_boot_df))) # Transpose of a list
std_boot_df=pd.DataFrame(std_boot_df,columns=['Feature','Actual stdev','CI stdev', 'Lower Bound', 'Upper Bound']) # List converted to dataframe
print("90% bootsrap confidence interval for the standard deviation of each feature-")
print("")
std_boot_df


# ### iv. Use your judgement to select the three most important time-domain features.

# The three most important time-domain features are-
# <ol>
#     <li>min</li>
#     <li>max</li>
#     <li>mean</li>
# </ol>

# ## (d) Binary Classification Using Logistic Regression

# ### i. Binary Classification-Bending vs. Other

#Stats table for raw features of train data

args1=[]
args2=[]
for i in range(1,7,1):
    args1.append('Min'+str(i))
    args1.append('Max'+str(i))
    args1.append('Mean'+str(i))
    args1.append('Median'+str(i))
    args1.append('StdDev'+str(i))
    args1.append('FirstQuartile'+str(i))
    args1.append('ThirdQuartile'+str(i))

minlist=[]
maxlist=[]
meanlist=[]
medianlist=[]
stdlist=[]
fqlist=[]
tqlist=[]

minlist=calc_min(df_train_list)
maxlist=calc_max(df_train_list)
meanlist=calc_mean(df_train_list)
medianlist=calc_median(df_train_list)
stdlist=calc_std(df_train_list)
fqlist=calc_fq(df_train_list)
tqlist=calc_tq(df_train_list)

args2.append(minlist)
args2.append(maxlist)
args2.append(meanlist)
args2.append(medianlist)
args2.append(stdlist)
args2.append(fqlist)
args2.append(tqlist)
statsDF_raw_train=print_stats_table(args1,args2)
statsDF_raw_train

#Stats table for all time series all normalized features of train data using MinMaxScaler

statsDF_norm_train=norm_ft(statsDF_raw_train,statsDF_raw_train)
statsDF_norm_train

#Stats table for time series 1,2,6 for normalized features min, mean, max

args1=[]
index=[0,1,5]
for i in index:
    args1.append('Min'+str(i+1))
    args1.append('Max'+str(i+1))
    args1.append('Mean'+str(i+1))

df_mmm_train=pd.DataFrame()
for col in args1:
    df_mmm_train[col]=statsDF_norm_train[col]

df_mmm_train


# ### Depict scatter plots of the features you specified in 1(c)iv extracted from time series 1, 2, and 6 of each instance, and use color to distinguish bending vs. other activities.

# Function to draw Scatterplot

def scatterplot(dataframe):
    a=0

    colname = []
    for i in range(len(dataframe.columns)-1):
        colname.extend(["",dataframe.columns[i]])

    def diagfunc(x, **kws):
        nonlocal a
        ax = plt.gca()
        ax.annotate(colname[a], xy=(0.5, 0.5), xycoords=ax.transAxes)
        a=a+1

    sns.set(context="paper")

    g = sns.PairGrid(dataframe,hue="y", height=3, vars=dataframe.columns[:-1]).map_diag(diagfunc)
    g = g.map_offdiag(plt.scatter)
    g = g.add_legend()

    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')

# Draw scatterplot of features min, max, mean extracted from time series 1, 2, 6

df_mmm_train['y']=train_target
scatterplot(df_mmm_train)

#Stats table for time series 3,4,5 for normalized features min, mean, max

args1=[]
index=[2,3,4]
for i in index:
    args1.append('Min'+str(i+1))
    args1.append('Max'+str(i+1))
    args1.append('Mean'+str(i+1))

df_mmm_train2=pd.DataFrame()
for col in args1:
    df_mmm_train2[col]=statsDF_norm_train[col]

df_mmm_train2

# Draw scatterplot of features min, max, mean extracted from time series 3, 4, 5

df_mmm_train2['y']=train_target
scatterplot(df_mmm_train2)


# ### ii. Break each time series in your training set into two (approximately) equal length time series.

# Function to split the dataset into number of parts
def divide_dataset(dataframe_list,lvalue):

    # new data column list
    collist = []
    for i in range(1,6*lvalue+1):
        collist.extend(["Min"+str(i),"Max"+str(i),"Mean"+str(i),"Median"+str(i),"StdDev"+str(i),"FirstQuartile"+str(i),"ThirdQuartile"+str(i)])

    # Extracting features from data and creating new data
    statsDF_split_train = []
    for df in dataframe_list:
        temp = []
        div_points=[-1]
        numerator = len(df)
        denominator = lvalue
        for l in range(lvalue):
            ans = int(numerator/denominator)
            div_points.append(div_points[l]+ans)
            numerator = numerator - ans
            denominator = denominator - 1

        df_split_list = [df.iloc[div_points[d]+1:div_points[d+1]] for d in range(lvalue)]
        for df_split in df_split_list:
            minlist = df_split.min() # Minimun
            maxlist = df_split.max() # Maximum
            meanlist = df_split.mean() # Mean
            medianlist = df_split.median() # Median
            stdlist = df_split.std() # Standard Deviation
            fqlist = df_split.quantile(0.25) # First Quartile
            tqlist = df_split.quantile(0.75) # Third Quartile
            for index in range(1,7):
                temp.extend([minlist[index],maxlist[index],meanlist[index],medianlist[index],stdlist[index],fqlist[index],tqlist[index]])
        statsDF_split_train.append(temp)

    return pd.DataFrame(statsDF_split_train, columns=collist)


# To show that the divide_dataset works properly

print(divide_dataset(df_train_list,2).shape)
print(divide_dataset(df_train_list,3).shape)
print(divide_dataset(df_train_list,7).shape)
print(divide_dataset(df_train_list,20).shape)

# Break train data into 2 parts

statsDF_train_split=divide_dataset(df_train_list,2)
statsDF_train_split

#Stats table for all time series all normalized features of train data using MinMaxScaler

statsDF_train_split_norm=norm_ft(statsDF_train_split,statsDF_train_split)
statsDF_train_split_norm

#Stats table for time series 1,2,12 for normalized features min, mean, max

args1=[]
index=[0,1,11]
for i in index:
    args1.append('Min'+str(i+1))
    args1.append('Max'+str(i+1))
    args1.append('Mean'+str(i+1))

df_mmm_train_split_norm=pd.DataFrame()
for col in args1:
    df_mmm_train_split_norm[col]=statsDF_train_split_norm[col]

df_mmm_train_split_norm

df_mmm_train_split_norm['y']=train_target
scatterplot(df_mmm_train_split_norm)

# ### iii. Break each time series in your training set into l &isin; &#123;1, 2, . . . , 20&#125; time series of approximately equal length and use logistic regression to solve the binary classification problem, using time-domain features.

# Logistic Regression to solve binary classification problem

# For l=1 to 20, and all features

acc = []
l_acc_df=[]

lval=[]

for l in range(1,21):
    d3traindata = divide_dataset(df_train_list,l) # creating DataFrame of training data
    d3testdata = divide_dataset(df_test_list,l) # creating DataFrame of test data

    df=divide_dataset(df_list,l)
    d3traindata_norm = norm_ft(df,d3traindata) # Normalizing train Data
    d3testdata_norm = norm_ft(df,d3testdata) # Normalizing test Data

    infinite = sys.maxsize
    logreg_model = LogisticRegression(C=infinite) # Logistic Regression Model Created
    logreg_model.fit(d3traindata_norm,train_target) # Fit the model on normalized train data

    acc.append(logreg_model.score(d3testdata_norm, test_target))

print("Best l value is",(1+acc.index(max(acc))))
print("Accuracy is {}%".format(max(acc)*100))

for i in range(1, 21):
    lval.append(i)

l_acc_df.append(lval)
l_acc_df.append(acc)

l_acc_df=list(map(list,zip(*l_acc_df)))
l_acc_df=pd.DataFrame(l_acc_df, columns=['l Value', 'Test Accuracy'])
l_acc_df

acclist = [] # list to store accuracy
lplist = [] # list to store (l,p) pairs

for l in range(1,21): # For number of divisions l=1 to 20
    d3_train_data = divide_dataset(df_train_list,l) # creating DataFrame of training data
    d3_train_data_norm = norm_ft(divide_dataset(df_list,l),d3_train_data) # Normalizing Data

    # For every p value
    for p in range(1,len(d3_train_data_norm.columns)+1): # For all combinations of p
        infinite=sys.maxsize # Inverse of regularization strength
        log_reg_model = LogisticRegression(C = infinite) # Logistic Regression Model created

        rfe = RFE(log_reg_model, p) # Select p features using recursive feature elimination
        rfe = rfe.fit(d3_train_data_norm,train_target) # Fit the model on normalized train data

        d3_train_data_norm_copy = d3_train_data_norm.copy() # Copy DataFrame to new DataFrame
        col = d3_train_data_norm_copy.columns

        # Drop all insignificant features
        selectorlist = rfe.support_.tolist() # Convert the supports to a list
        for sel in range(len(selectorlist)):
            if not selectorlist[sel]:
                d3_train_data_norm_copy = d3_train_data_norm_copy.drop(col[sel],axis = 1) # Drop the column if the value is false in the selectorlist

        cv_scores = cross_val_score(log_reg_model,d3_train_data_norm_copy,train_target,cv=5)
        acclist.append(cv_scores.mean())
        lplist.append([l,p])

maxacc_lplistindex=lplist[acclist.index(max(acclist))]
print("Best l value is",maxacc_lplistindex[0],"and best p value is",maxacc_lplistindex[1])
print("Accuracy(using cross-validation) is {}%".format(max(acclist)*100))


# ### Explain what the right way and the wrong way are to perform cross-validation in this problem.

# The right way to perform cross-validation is on the (l,p) pair.<br>
# The wrong way of performing cross-validation is for every l doing cv on p, or doing cv on l for every p.

# ### iv. Report the confusion matrix and show the ROC and AUC for your classifier on train data.

# Running logistic regression for l=1, p=7

d4_train_data = divide_dataset(df_train_list,1) # creating DataFrame of training data
d4_test_data = divide_dataset(df_test_list,1) # creating DataFrame of testing data
d4_train_data_norm = norm_ft(divide_dataset(df_list,1), d4_train_data) # Normalizing train Data
d4_test_data_norm = norm_ft(divide_dataset(df_list,1), d4_test_data) # Normalizing test Data

infinite=sys.maxsize # Inverse of regularization strength
d4_log_reg_model = LogisticRegression(C = infinite) # Logistic Regression Model created

d4_rfe = RFE(d4_log_reg_model, 7) # Select p features using recursive feature elimination
d4_rfe = d4_rfe.fit(d4_train_data_norm,train_target)

col=d4_train_data_norm.columns
selectorlist = d4_rfe.support_.tolist() # Convert the supports to a list
for sel in range(len(selectorlist)):
    if not selectorlist[sel]:
        d4_train_data_norm = d4_train_data_norm.drop(col[sel],axis = 1) # Drop the column if the value is false in the selectorlist
        d4_test_data_norm = d4_test_data_norm.drop(col[sel],axis = 1)

d4_log_reg_model.fit(d4_train_data_norm,train_target)
d4_predictions=d4_log_reg_model.predict(d4_test_data_norm)
cv_scores = cross_val_score(d4_log_reg_model,d4_train_data_norm,train_target,cv=5)
cv_acc=cv_scores.mean()
accuracy = d4_log_reg_model.score(d4_test_data_norm,test_target)

# (d) v. Compare the accuracy on the test set with the cross-validation accuracy you obtained previously
print("Test accuracy is {}%".format(accuracy*100))
print("Cross validation accuracy is {}%".format(cv_acc*100))


# ### Confusion matrix

d4_cm = confusion_matrix(test_target,d4_predictions)
df_cm = pd.DataFrame(d4_cm, index = ["Bending","Other"],
                  columns = ["Bending","Other"])
plt.figure(figsize = (9,7))
x = sns.heatmap(df_cm, annot=True)


# ### ROC and AUC for your classifier on train data

def convert_to_binary_labels(test_target):
    new_test_target=[]
    for z in test_target:
        if(z=='bending'):
            new_test_target.append(1)
        else:
            new_test_target.append(0)
    return new_test_target

# ROC and AUC

d4_fpr = dict()
d4_tpr = dict()
d4_roc_auc = dict()
classes_bin=['other','bending']
d4_fpr, d4_tpr, d4_threshold = roc_curve(convert_to_binary_labels(test_target), convert_to_binary_labels(d4_predictions))
d4_roc_auc = auc(d4_fpr, d4_tpr)

plt.plot(d4_fpr, d4_tpr,label='ROC curve (area = %0.2f)' % d4_roc_auc)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC for multi-class')
plt.legend(loc="lower right")
plt.show()


# ### Report the parameters of your logistic regression &beta;<sub>i</sub>s as well as the p-values associated with them

def calc_stats(model,X,y):
    #### Get p-values for the fitted model ####
    denom = (2.0*(1.0+np.cosh(model.decision_function(X))))
    denom = np.tile(denom,(X.shape[1],1)).T
    F_ij = np.dot((X/denom).T,X) ## Fisher Information Matrix
    Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
    sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
    z_scores = model.coef_[0]/sigma_estimates # z-score for each model coefficient
    p_values = [stat.norm.sf(abs(x))*2 for x in z_scores] ### two tailed test for p-values
    coef = model.coef_[0]
    return coef, sigma_estimates, z_scores, p_values

coeffs, se, zsc, pval = calc_stats(d4_log_reg_model, d4_train_data_norm, train_target)
print("Coefficient of intercept(beta0) is", d4_log_reg_model.intercept_[0])

params=[]
coefflist=[]
selist=[]
zlist=[]
plist=[]

features=d4_test_data_norm.columns

for i in range(len(d4_test_data_norm.columns)):
    coefflist.append(coeffs[i])
    selist.append(se[i])
    zlist.append(zsc[i])
    plist.append(pval[i])
params.append(features)
params.append(coefflist)
params.append(selist)
params.append(zlist)
params.append(plist)
params=list(map(list,zip(*params)))
paramsDF=pd.DataFrame(params, columns=['Features', 'coeff', 'std err', 'Z Score', 'P'])
paramsDF


# ### v. Compare the accuracy on the test set with the cross-validation accuracy you obtained previously

print("Test accuracy is {}%".format(accuracy*100))
print("Cross validation accuracy is {}%".format(cv_acc*100))


# ### vi. Do your classes seem to be well-separated to cause instability in calculating logistic regression parameters?

# Yes. The classes are linearly separable and hence there is an instability in calculating logistic regression parameters.

# ### vii. From the confusion matrices you obtained, do you see imbalanced classes?

# Yes, there is a class imbalance.<br>
# Class 1(Bending) has very few samples as compared to Class 0(Non-Bending) in the test dataaset.

# ### If yes, build a logistic regression model based on case-control sampling and adjust its parameters.

d7_train_data = divide_dataset(df_train_list,1) # creating DataFrame of training data
d7_test_data = divide_dataset(df_test_list,1) # creating DataFrame of testing data
d7_train_data_norm = norm_ft(divide_dataset(df_list,1), d7_train_data) # Normalizing train Data
d7_test_data_norm = norm_ft(divide_dataset(df_list,1), d7_test_data) # Normalizing test Data

infinite=sys.maxsize # Inverse of regularization strength
d7_log_reg_model = LogisticRegression(C = infinite, class_weight='balanced') # Logistic Regression Model created

d7_log_reg_model.fit(d7_train_data_norm,train_target)
d7_predictions=d7_log_reg_model.predict(d7_test_data_norm)

accuracy = d7_log_reg_model.score(d7_test_data_norm,test_target)

print("Test accuracy is {}%".format(accuracy*100))


# ### Report the confusion matrix, ROC, and AUC of the model.

d7_cm = confusion_matrix(test_target,d7_predictions)
df_cm = pd.DataFrame(d7_cm, index = ["Bending","Other"],
                  columns = ["Bending","Other"])
plt.figure(figsize = (9,7))
x = sns.heatmap(df_cm, annot=True)

# ROC and AUC of Logistic Regression model based on case-control sampling

fpr, tpr, threshhold = roc_curve(convert_to_binary_labels(test_target), convert_to_binary_labels(d7_predictions))
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.035, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC')
plt.legend(loc="lower right")
plt.show()


# ## (e) Binary Classification Using L1-penalized logistic regression

# ### i. Repeat 1(d)iii using L1-penalized logistic regression

# Two times CV
# For l=1 to 20, all features

lc = []
acc_lrcv = []

for l in range(1,21):
    print("l=",l)
    e_train_data = divide_dataset(df_train_list,l) # creating DataFrame of training data

    df=divide_dataset(df_list,l)
    e_train_data_norm = norm_ft(df,e_train_data) # Normalizing Data
    cval=0.0001
    while(cval<10000):
        e_log_reg_cv_model = LogisticRegression(C=cval, solver='liblinear', penalty = 'l1') # Logistic Regression Model Created
        e_log_reg_cv_model.fit(e_train_data_norm,train_target)
        crossval_scores = cross_val_score(e_log_reg_cv_model,e_train_data_norm,train_target,cv=5)
        lc.append([l,cval])
        acc_lrcv.append(crossval_scores.mean())
        cval=cval*10
print("Best l value is",lc[acc_lrcv.index(max(acc_lrcv))][0],"and best C value is",lc[acc_lrcv.index(max(acc_lrcv))][1])
print("Accuracy is {}%".format(max(acc_lrcv)*100))

# For l=1, C=100
e2_train_data = divide_dataset(df_train_list,1) # creating DataFrame of training data
e2_test_data = divide_dataset(df_test_list,1) # creating DataFrame of testing data

df=divide_dataset(df_list,1)
e2_train_data_norm = norm_ft(df,e2_train_data) # Normalizing train Data
e2_test_data_norm = norm_ft(df,e2_test_data) # Normalizing test Data

e2_log_reg_cv_model = LogisticRegression(C=100, solver='liblinear', penalty = 'l1') # Logistic Regression Model Created
e2_log_reg_cv_model.fit(e2_train_data_norm,train_target)
test_acc_lrcv=e2_log_reg_cv_model.score(e2_test_data_norm,test_target)
print("Test Accuracy is {}%".format(test_acc_lrcv*100))


# ## ii. Compare the L1-penalized with variable selection using p-values. Which one performs better? Which one is easier to implement?

# The train accuracy of L1-penalized Logistic Regression model is 97.14285714285715% and the test accuracy is 100%.<br>
# The train and test accuracy of the model doing variable selection using p values is 100%.<br>
# Thus, model doing variable selection using p values performs better than the other.<br>
# L1-penalized Logistic Regression model is easier to implement.<br>

# ## (f) Multi-class Classification (The Realistic Case)

# ### i. Find the best l in the same way as you found it in 1(e)i to build an L1-penalized multinomial regression model to classify all activities in your training set.

# For l=1 to 20, all features

f_lc = []
f_acc_lrcv = []

for l in range(1,21):
    print("l=",l)
    f_train_data = divide_dataset(df_train_list,l) # creating DataFrame of training data
    f_test_data = divide_dataset(df_test_list,l)

    df=divide_dataset(df_list,l)
    f_train_data_norm = norm_ft(df,f_train_data) # Normalizing Data
    f_test_data_norm = norm_ft(df,f_test_data)
    f_cval=0.00001
    while(f_cval<10000):
        f_log_reg_cv_model = LogisticRegression(C=f_cval, solver='saga', penalty='l1', multi_class='multinomial') # Logistic Regression Model Created
        crossval_scores = cross_val_score(f_log_reg_cv_model,f_train_data_norm,train_all_target,cv=5)
        f_lc.append([l,f_cval])
        f_acc_lrcv.append(crossval_scores.mean())
        f_cval=f_cval*10

f_bestlc = f_lc[f_acc_lrcv.index(max(f_acc_lrcv))]
print("Best l value is",f_bestlc[0],"and best C value is",f_bestlc[1])
print("Accuracy is {}%".format(max(f_acc_lrcv)*100))


# ### Report your test error.

f_train_data = divide_dataset(df_train_list,f_bestlc[0]) # creating DataFrame of training data
f_test_data = divide_dataset(df_test_list,f_bestlc[0]) # creating DataFrame of testing data

df=divide_dataset(df_list,f_bestlc[0])
f_train_data_norm = norm_ft(df,f_train_data) # Normalizing train Data
f_test_data_norm = norm_ft(df,f_test_data) # Normalizing test Data

f1_log_reg_cv_model = LogisticRegression(C=f_bestlc[1], solver='saga', penalty='l1', multi_class='multinomial').fit(f_train_data_norm, train_all_target) # Logistic Regression Model Created

# Report test error
f_predictions=f1_log_reg_cv_model.predict(f_test_data_norm)
f_pred=f1_log_reg_cv_model.decision_function(f_test_data_norm)

f1_train_acc=f1_log_reg_cv_model.score(f_train_data_norm, train_all_target)
f_train_error = 1-f1_train_acc
f1_acc=f1_log_reg_cv_model.score(f_test_data_norm, test_all_target)
f_test_error = 1-f1_acc
print("Train Accuracy of L1 penalized multinomial regression model is {}%".format(f1_train_acc*100))
print("The Test Error is ",f_train_error)
print("Test Accuracy of L1 penalized multinomial regression model is {}%".format(f1_acc*100))
print("The Test Error is ",f_test_error)


# ### Confusion matrix for L1-penalized multinomial regression model

f1_cm = confusion_matrix(test_all_target, f_predictions)
classes=["bending1","bending2", "cycling", "lying", "sitting", "standing", "walking"]
df_cm = pd.DataFrame(f1_cm, index = classes,
                  columns = classes)
plt.figure(figsize = (9,7))
x = sns.heatmap(df_cm, annot=True)

# ROC and AUC

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], threshold = roc_curve(label_binarize(test_all_target, classes=classes)[:,i], f_pred[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i],
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC for multi-class')
plt.legend(loc="lower right")
plt.show()


# ### ii. Repeat 1(f)i using a Naive Bayes' classifier. Use both Gaussian and Multinomial priors and compare the results.

# ### Gaussian Naive Bayes

f2_gnb_cv_acc=[]

for l in range(1,21):
    f_train_data = divide_dataset(df_train_list,l) # creating DataFrame of training data
    f_test_data = divide_dataset(df_test_list,l) # creating DataFrame of testing data

    gnb = GaussianNB()
    gnb.fit(f_train_data, train_all_target)

    f2_gnb_cv_scores = cross_val_score(gnb,f_train_data, train_all_target,cv=5)
    f2_gnb_cv_acc.append(f2_gnb_cv_scores.mean())

f2_bestl=1+f2_gnb_cv_acc.index(max(f2_gnb_cv_acc))
print("Best l value is", f2_bestl)
print("Cross validation Accuracy of Gaussian Naive Bayes is {}%".format(max(f2_gnb_cv_acc)*100))

# for l=2
f_gnb_train_data = divide_dataset(df_train_list, f2_bestl) # creating DataFrame of training data
f_gnb_test_data = divide_dataset(df_test_list, f2_bestl)
gnb = GaussianNB()
gnb.fit(f_gnb_train_data, train_all_target)
gnb_train_acc=gnb.score(f_gnb_train_data, train_all_target)
gnb_test_acc=gnb.score(f_gnb_test_data, test_all_target)
gnb_test_error=1-gnb_test_acc

print("Train accuracy is {}%".format(gnb_train_acc*100))
print("Test accuracy is {}%".format(gnb_test_acc*100))
print("Test Error is", gnb_test_error)

f_train_data = divide_dataset(df_train_list,f2_bestl) # creating DataFrame of training data
f_test_data = divide_dataset(df_test_list,f2_bestl) # creating DataFrame of testing data
gnb = GaussianNB()
gnb.fit(f_train_data, train_all_target)
f2_gnb_predictions=gnb.fit(f_train_data, train_all_target).predict(f_test_data)
f2_gnb_pred=gnb.predict_proba(f_test_data)

f2_gnb_cm = confusion_matrix(test_all_target, f2_gnb_predictions)
classes=["bending1","bending2", "cycling", "lying", "sitting", "standing", "walking"]
df_cm = pd.DataFrame(f2_gnb_cm, index = classes,
                  columns = classes)
plt.figure(figsize = (9,7))
x = sns.heatmap(df_cm, annot=True)

# ROC and AUC

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], threshold = roc_curve(label_binarize(test_all_target, classes=classes)[:,i], f2_gnb_pred[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i],
             label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC for multi-class Gaussian Naive Bayes classifier')
plt.legend(loc="lower right")
plt.show()


# ### Mulltinomial Naive Bayes
f2_mul_cv_acc=[]

for l in range(1,21):
    f_train_data = divide_dataset(df_train_list,l) # creating DataFrame of training data
    f_test_data = divide_dataset(df_test_list,l) # creating DataFrame of testing data
    clf = MultinomialNB()
    clf.fit(f_train_data, train_all_target)

    # Cross-validation
    f2_mul_cv_scores = cross_val_score(clf,f_train_data, train_all_target,cv=5)
    f2_mul_cv_acc.append(f2_mul_cv_scores.mean())

f2_mul_bestl=1+f2_mul_cv_acc.index(max(f2_mul_cv_acc))
print("Best l value is", f2_mul_bestl)
print("Cross-validation accuracy of Multinomial Naive Bayes is {}%".format(max(f2_mul_cv_acc)*100))

# for l=1(best l)
f_mul_train_data = divide_dataset(df_train_list, f2_mul_bestl) # creating DataFrame of training data
f_mul_test_data = divide_dataset(df_test_list, f2_mul_bestl)
clf = MultinomialNB()
clf.fit(f_mul_train_data, train_all_target)
mul_train_acc=clf.score(f_mul_train_data, train_all_target)
mul_test_acc=clf.score(f_mul_test_data, test_all_target)
mul_test_error=1-mul_test_acc

print("Train accuracy is {}%".format(mul_train_acc*100))
print("Test accuracy is {}%".format(mul_test_acc*100))
print("Test Error is",mul_test_error)

f_train_data = divide_dataset(df_train_list,f2_mul_bestl) # creating DataFrame of training data
f_test_data = divide_dataset(df_test_list,f2_mul_bestl) # creating DataFrame of testing data
clf = MultinomialNB()
clf.fit(f_train_data, train_all_target)
f2_mul_predictions=clf.fit(f_train_data, train_all_target).predict(f_test_data)
f2_mul_pred=clf.predict_proba(f_test_data)

f2_mul_cm = confusion_matrix(test_all_target, f2_mul_predictions)
classes=["bending1","bending2", "cycling", "lying", "sitting", "standing", "walking"]
df_cm = pd.DataFrame(f2_mul_cm, index = classes,
                  columns = classes)
plt.figure(figsize = (9,7))
x = sns.heatmap(df_cm, annot=True)

# ROC and AUC

mul_fpr = dict()
mul_tpr = dict()
mul_roc_auc = dict()
for i in range(len(classes)):
    mul_fpr[i], mul_tpr[i], threshold = roc_curve(label_binarize(test_all_target, classes=classes)[:,i], f2_mul_pred[:,i])
    mul_roc_auc[i] = auc(mul_fpr[i], mul_tpr[i])

for i in range(len(classes)):
    plt.plot(mul_fpr[i], mul_tpr[i],
             label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], mul_roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC for multi-class Multinomial Naive Bayes classifier')
plt.legend(loc="lower right")
plt.show()


# Gaussian Naive Bayes-<br>
# Train Accuracy=100.0%<br>
# Test Accuracy=63.1578947368421%<br>
# <br>
# Multinomial Naive Bayes-<br>
# Train Accuracy=92.7536231884058%<br>
# Test Accuracy=89.47368421052632%<br>

# The Gaussian Naive Bayes Classifier gives better train accuracy(100%) than the Multinomial Naive Bayes classifier(92.7536231884058%).<br>
# The Multinomial Naive Bayes Classifier gives better test accuracy(89.47368421052632%) than the Gaussian Naive Bayes classifier(63.1578947368421%).

# ### iii. Which method is better for multi-class classification in this problem?

table=[]
model_name=['L1-penalized', 'Gaussian NB', 'Multinomial NB']
accuracy_train=[]
accuracy_test=[]
accuracy_train.append(f1_train_acc*100)
accuracy_train.append(gnb_train_acc*100)
accuracy_train.append(mul_train_acc*100)
accuracy_test.append(f1_acc*100)
accuracy_test.append(gnb_test_acc*100)
accuracy_test.append(mul_test_acc*100)
table.append(model_name)
table.append(accuracy_train)
table.append(accuracy_test)
table=list(map(list,zip(*table)))
tableDF=pd.DataFrame(table, columns=['Model','Train Accuracy','Test Accuracy'])
tableDF


# Multinomial Naive Bayes gives better test accuracy, and pretty good train accuracy.
