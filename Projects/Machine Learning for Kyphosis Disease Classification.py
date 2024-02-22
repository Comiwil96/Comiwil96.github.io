#!/usr/bin/env python
# coding: utf-8

# # TASK 1: UNDERSTAND THE PROBLEM STATEMENT

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # TASK #2: IMPORT LIBRARIES AND DATASETS
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# from jupyterthemes import jtplot
# jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 


# In[3]:


df = pd.read_csv('Kyphosis.csv')
df


# In[4]:


df.isnull().sum()/len(df)*100


# In[5]:


df.dtypes


# In[6]:


df['Age'] = df['Age'].apply(lambda x: x / 12)
df


# **PRACTICE OPPORTUNITY #1 [OPTIONAL]:**
# - **List the average, minimum and maximum age (in years) considered in this study using two different methods**

# In[7]:


print(df.describe())

print('The average age of the child receiving corrective spinal surgery is:', df['Age'].mean())
print('The oldest child to receive corrective spinal surgery is:', df['Age'].max())
print('The youngest child to receive corrective spinal surgery is:', df['Age'].min())


# # TASK #3: PERFORM DATA VISUALIZATION

# In[8]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_y = LabelEncoder()
df['Kyphosis'] = LabelEncoder_y.fit_transform(df['Kyphosis'])


# In[9]:


Kyphosis_df = df
Kyphosis_df


# In[10]:


Kyphosis_True = Kyphosis_df[Kyphosis_df['Kyphosis']==1]


# In[11]:


Kyphosis_False = Kyphosis_df[Kyphosis_df['Kyphosis']==0]


# In[12]:


print( 'Disease present after operation percentage =', (len(Kyphosis_True) / len(Kyphosis_df) )*100,"%")


# In[13]:


plt.figure(figsize=(10,10)) 
sns.heatmap(Kyphosis_df.corr(), annot = True)


# In[14]:


sns.pairplot(Kyphosis_df, hue = 'Kyphosis')


# **PRACTICE OPPORTUNITY #2 [OPTIONAL]:**
# - **Plot the data countplot showing how many samples belong to each class**

# In[15]:


sns.countplot(x=Kyphosis_df['Kyphosis'])


# # TASK #4: CREATE TESTING AND TRAINING DATASET/DATA CLEANING

# In[16]:


# Let's drop the target label coloumns

X = Kyphosis_df.drop(columns=['Kyphosis'], axis=1)
y = Kyphosis_df['Kyphosis']


# In[17]:


X


# In[18]:


y


# In[19]:


from sklearn.model_selection import train_test_split


# In[40]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
x_train.shape


# In[21]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# # TASK #5: TRAIN A LOGISTIC REGRESSION CLASSIFIER MODEL

# ![image.png](attachment:image.png)

# In[22]:


x_train.shape


# In[23]:


y_train.shape


# In[24]:


x_test.shape


# In[25]:


y_test.shape


# In[26]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)


# # TASK #6: EVALUATE TRAINED MODEL PERFORMANCE 

# In[27]:


from sklearn.metrics import classification_report, confusion_matrix


# In[42]:


# Predicting the Test set results
yhat = lr.predict(x_test)
cm = confusion_matrix(y_test, yhat)
sns.heatmap(cm, annot = True)


# In[43]:


print(classification_report(y_test, yhat))


# # TASK #7: UNDERSTAND THE THEORY AND INTUITION BEHIND DECISION TREES AND RANDOM FOREST CLASSIFIER MODELS

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # TASK #8: IMPROVE THE MODEL 

# In[35]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)


# In[36]:


# Predicting the Test set results
y_predict_test = decision_tree.predict(x_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[37]:


print(classification_report(y_test, y_predict_test))


# In[44]:


feature_importances = pd.DataFrame(decision_tree.feature_importances_,
                                   index = x_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)

print(feature_importances)


# **PRACTICE OPPORTUNITY #3 [OPTIONAL]:**
# - **Train a random forest classifier model and assess its performance**
# - **Plot the confusion matrix**
# - **Print the classification Report**
# 

# In[46]:


# create a random forest classifier object
decision_tree.fit(x_train, y_train)
# predict the test results
yhat1 = decision_tree.predict(x_test)
cm = confusion_matrix(yhat, y_test)
sns.heatmap(cm, annot=True)
# print classification report
print(classification_report(y_test, y_predict_test))


# # GREAT JOB! 

# # PRACTICE OPPORTUNITIES SOLUTIONS

# **PRACTICE OPPORTUNITY #1 SOLUTION:**
# - **List the average, minimum and maximum age (in years) considered in this study using two different methods**

# In[ ]:


Kyphosis_df.describe()


# In[ ]:


Kyphosis_df['Age'].mean()/12


# In[ ]:


Kyphosis_df['Age'].min()/12


# In[ ]:


Kyphosis_df['Age'].max()/12


# **PRACTICE OPPORTUNITY #2 SOLUTION:**
# - **Plot the data countplot showing how many samples belong to each class**

# In[ ]:


sns.countplot(x = Kyphosis_df['Kyphosis'], label = "Count");


# **PRACTICE OPPORTUNITY #3 SOLUTION:**
# - **Train a random forest classifier model and assess its performance**
# - **Plot the confusion matrix**
# - **Print the classification Report**
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier()
RandomForest.fit(X_train, y_train)

# Predicting the Test set results
y_predict_test = RandomForest.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict_test))


# In[ ]:




