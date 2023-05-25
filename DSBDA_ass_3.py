#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("Iris.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.mean()


# In[7]:


df.min()


# In[8]:


df.max()


# In[9]:


df.std()


# In[10]:


df.var()


# In[11]:


df.median()


# In[12]:


df.mode()


# In[14]:


df.groupby(['Species']).SepalLengthCm.agg(['min','max','mean','median','std','var'])


# In[15]:


df.groupby(['Species']).SepalWidthCm.agg(['min','max','mean','median','std','var'])


# In[16]:


df.groupby(['Species']).PetalLengthCm.agg(['min','max','mean','median','std','var'])


# In[17]:


df.groupby(['Species']).PetalWidthCm.agg(['min','max','mean','median','std','var'])


# In[ ]:




