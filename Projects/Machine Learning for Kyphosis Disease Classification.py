#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Kyphosis.csv')
df

df.isnull().sum()/len(df)*100 # check to see if any values in the columns have a null or NaN value
df.dtypes

df['Age'] = df['Age'].apply(lambda x: x / 12) # convert the age from months to years
print(df.describe()) # basic descriptive statistics analysis

# import Scikit learn for machine learning model building
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_y = LabelEncoder()
df['Kyphosis'] = LabelEncoder_y.fit_transform(df['Kyphosis']) # turn binary answers into numbers
Kyphosis_df = df # create a Kyphosis_df dataframe for more specific terminology
Kyphosis_df
Kyphosis_True = Kyphosis_df[Kyphosis_df['Kyphosis']==1] # set Kyphosis values to objects
Kyphosis_False = Kyphosis_df[Kyphosis_df['Kyphosis']==0]
print( 'Disease present after operation percentage =', (len(Kyphosis_True) / len(Kyphosis_df) )*100,"%")

# create a plot to examine the correlational data
plt.figure(figsize=(10,10)) 
sns.heatmap(Kyphosis_df.corr(), annot = True)
sns.pairplot(Kyphosis_df, hue = 'Kyphosis')
sns.countplot(x=Kyphosis_df['Kyphosis'])

# creating testing data
X = Kyphosis_df.drop(columns=['Kyphosis'], axis=1)
y = Kyphosis_df['Kyphosis']
print(X)
print(y)

# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
x_train.shape

# normalize the training and testing data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# create logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train) # fit data to the logistic regression model

# evaluate model performance
from sklearn.metrics import classification_report, confusion_matrix

# Predicting the Test set results
yhat = lr.predict(x_test)
cm = confusion_matrix(y_test, yhat)
sns.heatmap(cm, annot = True)
print(classification_report(y_test, yhat))

# import decision tree classifier and fit training data
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)

# predict the Test set results
y_predict_test = decision_tree.predict(x_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict_test))

# examine the most important features per the decision tree
feature_importances = pd.DataFrame(decision_tree.feature_importances_,
                                   index = x_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)




