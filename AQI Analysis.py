#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis of Air Quality Index of Delhi

# Air performs a vital role in supporting life on Earth. But with the passage of time the fresh and pure air is gradually    getting contaminated due to increase in air pollution.

# Problem of air pollution is increasingly getting more serious. Increasing levels of pollutants in air is causing extreme health disorder. It directly affects a population of millions who are suffering from shortness of breath, eye irritation to chronic respiratory disorders, pneumonia, acute asthma etc.

# This study is based on hourly data collected for Delhi state (www.cpcb.nic.in)

# We will be doing Exploratory Data Analysis for this data using pandas, numpy, matplotlib and seaborn libraries.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("C:\\Users\\Ankita\\Desktop\\Datasets\\Delhi_AQI.csv")
print(df.head())
date=df.iloc[: ,1]
AQI_bucket=df.iloc[:,-1]


# ## Descriptive Statistics

# In[3]:


df.describe()


# Here we get PM10 pollutant with highest mean (232.8) which shows that for poor air quality of Delhi PM10 is the most responsible pollutant present in the air.

# ## Data Cleaning

# we have to clean the dataset to start the Exploratory Data Analysis.

# In[4]:


print(df.isnull().sum().plot(kind='bar'))


# In[5]:


print(df.shape)


# In[6]:


# Importing the SimpleImputer class
from sklearn.impute import SimpleImputer


# In[7]:


imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df1=df.iloc[: ,2:15]
df1= imputer.fit_transform(df1)
df1=pd.DataFrame(df1)
print(df1.isnull().sum().plot(kind='bar'))


# Our dataset is now clean with no null values.

# In[8]:


df1.loc[:,"Date"]=date
df1.loc[:,"AQI_bucket"]=AQI_bucket
df1.columns=['PM2.5','PM10','NO','NO2','NOX','NH3','CO','SO2','O3','Benzene','Toluene','Xylene','AQI','Date','AQI_bucket']
print(df1.head())


# ## EDA

# ## How AQI Bucket list is distributed?

# In[9]:


#AQI Bucket list we have
df1['AQI_bucket'].value_counts()


# In[10]:


sns.countplot(x=df.iloc[:,-1],data=df1)


# There are six AQI categories, namely Good, Satisfactory, moderately polluted, Poor, Very Poor, and Severe.

# Most of the observations lie under Moderate, Poor, Very Poor region ( which is serious matter of concern).

# Very few observations are falls under Good air quality, which means most of the time air of Delhi is highly polluted.

# ## How AQI is distributed?

# In[11]:


sns.distplot(df1['AQI'],kde=True,hist=True)


# In[12]:


df1.plot(kind='scatter',x='PM2.5',y='AQI')


# PM2.5 directly affecting the AQI value. 
# Higher the concentration of PM2.5 in the atmosphere poorer the quality of air is.
# PM2.5 contributing highly for the poor air quality of Delhi.

# In[13]:


df1.plot(kind='scatter',x='Benzene',y='AQI')


# Even if the Benzene concentration is low in the environment,the AQI value is high. So,we can say that Benzene does not affect much in the AQI

# In[14]:


sns.regplot(x='PM2.5',y='PM10',data=df1,scatter=True,fit_reg=True)


# PM10 and PM2.5 are correlated, as concentration of PM2.5 increases the concentraion of PM10 also increases.

# ### How each pollutant affects AQI ?

# In[15]:


fig,((x1,x2),(x3,x4),(x5,x6))=plt.subplots(nrows=3,ncols=2,figsize=(10,6))
sns.scatterplot(x='NO2',y='AQI',data=df1,color='red',ax=x1)
sns.scatterplot(x='NO',y='AQI',data=df1,color='green',ax=x2)
sns.scatterplot(x='NH3',y='AQI',data=df1,color='blue',ax=x3)
sns.scatterplot(x='CO',y='AQI',data=df1,color='orange',ax=x4)
sns.scatterplot(x='SO2',y='AQI',data=df1,color='pink',ax=x5)
sns.scatterplot(x='O3',y='AQI',data=df1,color='violet',ax=x6)


# From the above plots we find that NO2, NO and NH3 are positively correlated with AQI, means as the concentration of        these three pollutants increases in the enviroment, the air becomes more polluted.

# CO, SO2 and O3 are not affecting the Air Quality much.

# ## How Pollutants are correlated?

# In[20]:


df3=df1.iloc[:,0:12]
sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(data=df3.corr(), annot=True)


# From the above heatmap we find that PM2.5 and PM10 are highly correlated with correlation coefficient 0.85

#  CO, O3, Toulene and xylene are found to be poorely related( or we can say uncorrelated) with other pollutants.
