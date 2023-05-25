#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn


# In[2]:


from sklearn.datasets import load_diabetes


# In[3]:


df = load_diabetes()


# In[4]:


df


# In[5]:


import pandas as pd
import numpy as np


# In[6]:


df_x = pd.DataFrame(df.data,columns = df.feature_names)


# In[7]:


df_y = pd.DataFrame(df.target)


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.3)


# In[10]:


from sklearn import linear_model


# In[13]:


reg = linear_model.LinearRegression()


# In[14]:


reg.fit(x_train,y_train)


# In[15]:


y_pred = reg.predict(x_test)


# In[17]:


np.mean((y_test - y_pred)**2)


# In[ ]:




