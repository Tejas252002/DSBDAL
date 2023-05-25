#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


dataset = sns.load_dataset('titanic') 
dataset.head()


# In[4]:


dataset.info()


# In[5]:


dataset.shape


# In[6]:


sns.distplot(dataset['age'], bins = 10)


# In[7]:


sns.jointplot(dataset['age'], y = dataset['fare'], kind = 'scatter')


# In[8]:


sns.rugplot(dataset['fare'])


# In[9]:


sns.barplot(x='sex', y='age', data=dataset, estimator=np.std)


# In[10]:


sns.countplot(x='sex', data=dataset)


# In[11]:


sns.boxplot(x='sex', y='age', data=dataset)


# In[12]:


sns.violinplot(x='sex', y='age', data=dataset)


# In[4]:


import seaborn as sns


# In[6]:


dataset = sns.load_dataset('titanic') 


# In[7]:


sns.stripplot(x='sex', y='age', data=dataset,hue='alive')


# In[8]:


dataset.corr()


# In[10]:


corr = dataset.corr()
sns.heatmap(corr)


# In[ ]:




