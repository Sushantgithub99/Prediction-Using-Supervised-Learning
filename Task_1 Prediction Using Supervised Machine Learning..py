#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation
# # GRIP SEP 2023 
# Name :- SUSHANt SHEKHAR DASHPUTE
# 
# Domain:- Data Science & Business Analytics
# 
# Task_1= Prediction using supervised machine learning
# 
# Problem Statement:-
# predict the percentage of students based on No. of study hours    
#     

# In[13]:


#Importing all required libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from math import sqrt


# In[14]:


#Reading data from link 
url="http://bit.ly/w-data"
stu_data=pd.read_csv(url)

print("Data imported successfully")
stu_data.head(10)


# In[15]:


stu_data.describe()


# The Average study hours of students is 5.012
# 
# The Average score is 51.48

# In[16]:


# To check whether null values are present or not
stu_data.isnull().sum()


# From the output we can observe that there are no null values present in the data

# **Visualizing Data** :

# In[62]:




plt.rcParams["figure.figsize"]=[7.5,5]
plt.rcParams["figure.autolayout"]=True

boxplot=stu_data[["Hours","Scores"]].plot(kind="box",title="Boxplt of Study Hours & Student Scores",patch_artist=True
                                          ,boxprops={"facecolor":"skyblue","edgecolor":"black"})


# In[66]:


#Plotting the distribution of scores
stu_data.plot(x="Hours",y="Scores",style="o")
plt.title("Study Hours v/s Scores")
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.show()


# **Conclusion** : From the above graph we can see that there is highly positive linear relationship between study Hours and socres obtained by sudents

# # Checking Normality Of Data

# In[67]:


#By Graphical representation
#For Study_Hours
stats.probplot(stu_data["Hours"],plot=plt)


# In[69]:


#For Scores
stats.probplot(stu_data["Scores"],plot=plt)


# # Checking Normality by Statistical test

# Ho:Data is distributed Normally
#     
# H1:Data is not distributed normally
#     

# In[70]:


# We use shpairo-wilk normality test as there are less than 50 observations
stats.shapiro(stu_data["Hours"])


# In[71]:


stats.shapiro(stu_data["Scores"])


# **Conclusion**: 
# 
# 1] Hours = Here we can see that the p_value=0.149 which is greater than level of significance value (0.05) therefore we may accept Ho at 5% l.o.s and we can say that the data is normally distributed.
# 
# 2] Scores = Here we can see that the p_value=0.149 which is greater than level of significance value (0.01) therefore we may accept Ho at 1% l.o.s and we can say that the data is normally distributed. 

# # Correlation between Study Hours & Student Scores

# In[77]:


Correlation = stu_data.corr()
print(Correlation)

#visualizing correlation by heatmap

sns.heatmap(Correlation,annot=True)


# **Conclusion**:- Thus we may say that there is highly positive correlation between Study Hours & Student Scores

# # Preparing the data 

# In[78]:


x=stu_data.iloc[:, :-1].values
y=stu_data.iloc[:,1].values


# # Splitting data into training and testing :- 

# Here the independent variable is "Study Hours" and dependent variable is "Student Scores"
# 
# To make predictions about the scores obtained based on number of study hours we will run train test split method where we are going to divide our dataset in 8:2 ratio
# 
# Here 80% is the training dataset through which we are going to mae predictions .
# 

# In[79]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[80]:


reg= LinearRegression()
reg.fit(x_train,y_train)

print('Training Completed')


# In[87]:


# Plotting Regression line
line = reg.coef_*x+reg.intercept_

#plotting for the test data

plt.scatter(x,y,marker='+')
plt.plot(x,line,color='red')
plt.title("Study Hours v/s Student Scores")
plt.xlabel("Study Hours")
plt.ylabel("Student Scores")
plt.show()


# # Prediction

# In[86]:


print(x_test)


# # Actual V/s Predicted

# In[98]:


y_pre=reg.predict(x_test)
df = pd.DataFrame({ "Actual Value" : y_test , "Predicted Value" : y_pre})
df


# # Predicting the scores by user input

# In[100]:


value=float(input("Enter the Study Hours:"))
pred = reg.predict([[value]])
print("If the student Study",value,"Hours daily","then score will be ",pred)


# # Thank You
# 
