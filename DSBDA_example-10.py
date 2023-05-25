#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[3]:


df=pd.read_csv("Iris.csv") 
df


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[10]:


x = df["SepalLengthCm"]
plt.hist(x, bins = 20, color = "green") 
plt.title("Sepal Length in cm") 
plt.xlabel("SepalLengthCm") 
plt.ylabel("Count")


# In[11]:


x = df["PetalLengthCm"]
plt.hist(x, bins = 20, color = "green") 
plt.title("Petal Length in cm") 
plt.xlabel("PetalLengthCm") 
plt.ylabel("Count")


# In[12]:


x = df["SepalWidthCm"]
plt.hist(x, bins = 20, color = "green") 
plt.title("Sepal Width in cm") 
plt.xlabel("SepalWidthCm")
                                                                                  


# In[13]:


x = df["PetalWidthCm"]
plt.hist(x, bins = 20, color = "green") 
plt.title("Petal width in cm") 
plt.xlabel("PetalWidthCm") 
plt.ylabel("Count")


# In[14]:


x = df["Species"]
plt.hist(x, bins = 20, color = "green") 
plt.title("Species Distribution") 
plt.xlabel("Species") 
plt.ylabel("Count")


# In[15]:


sns.boxplot(df['SepalLengthCm'])


# In[16]:


sns.boxplot(df['SepalWidthCm'])


# In[17]:


sns.boxplot(df['PetalLengthCm'])


# In[18]:


sns.boxplot(df['PetalWidthCm'])


# In[19]:


species_to_idx={'setosa':0,'versicolor':1,'virginica':2} 
df.replace(species_to_idx, inplace=True)
df


# In[20]:


sns.boxplot(df['PetalWidthCm'])


# In[ ]:




