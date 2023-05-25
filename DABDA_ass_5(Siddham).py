#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('Social_Network_Ads.csv')


# In[3]:


df


# In[ ]:





# In[4]:


df_x = df.iloc[:,[2,3]].values


# In[5]:


df_x


# In[6]:


df_y = df.iloc[:,[4]].values


# In[7]:


df_y


# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


ss = StandardScaler()


# In[11]:


df_x = ss.fit_transform(df_x)


# In[12]:


df_x


# In[14]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size= 0.3)


# In[21]:


x_train


# In[17]:


from sklearn import linear_model


# In[18]:


reg = linear_model.LogisticRegression()


# In[22]:


reg.fit(x_train,y_train)


# In[23]:


y_pred = reg.predict(x_test)


# In[24]:


from sklearn.metrics import confusion_matrix


# In[25]:


cm = confusion_matrix(y_test,y_pred)


# In[26]:


cm


# In[27]:


tp=cm[0][0]


# In[28]:


tp


# In[33]:


tn=cm[1][1]


# In[34]:


tn


# In[35]:


fp=cm[0][1]


# In[36]:


fn = cm[1][0]


# In[37]:


fn


# In[38]:


accuracy = (tp+tn)/(tp+tn+fn+fp)


# In[39]:


accuracy


# In[43]:


error = (1 - accuracy)*100


# In[44]:


error


# In[45]:


precission = (tp)/(tp+fp)


# In[46]:


precission


# In[47]:


recall = tp/(tp+fn)


# In[48]:


recall


# In[ ]:




