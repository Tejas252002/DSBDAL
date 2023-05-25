#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# In[3]:


from seaborn import load_dataset 
data= sns.load_dataset('titanic') 
tips= sns.load_dataset('tips')


# In[4]:


data.head


# In[6]:


data.shape


# In[7]:


data.columns


# In[8]:


data.info()


# In[9]:


data.isnull().sum()


# In[10]:


data['age'] = data['age'].fillna(0) 
data.isnull().sum()


# In[11]:


data.describe


# In[12]:


sns.countplot(data['survived'])


# In[13]:


data['sex'].value_counts().plot(kind="pie", autopct="%.2f") 
plt.show()


# In[14]:


plt.hist(data['age'], bins=5) 
plt.show()


# In[15]:


sns.distplot(data['age'])
plt.show()


# In[16]:


sns.scatterplot(tips["total_bill"], tips["tip"])


# In[17]:


sns.scatterplot(tips["total_bill"], tips["tip"], hue=tips["sex"]) 
plt.show()


# In[18]:


sns.scatterplot(tips["total_bill"], tips["tip"], hue=tips["sex"], style=tips['smoker'])
plt.show()


# In[19]:


sns.barplot(data['pclass'], data['age'])
plt.show()


# In[20]:


sns.barplot(data['pclass'], data['fare'], hue = data["sex"]) 
plt.show()


# In[21]:


sns.boxplot(data['sex'], data["age"])


# In[22]:


sns.boxplot(data['sex'], data["age"], data["survived"])
plt.show()


# In[23]:


sns.distplot(data[data['survived'] == 0]['age'], hist=False, color="blue") 
sns.distplot(data[data['survived'] == 1]['age'], hist=False, color="orange") 
plt.show()


# In[24]:


pd.crosstab(data['pclass'], data['survived'])


# In[ ]:




