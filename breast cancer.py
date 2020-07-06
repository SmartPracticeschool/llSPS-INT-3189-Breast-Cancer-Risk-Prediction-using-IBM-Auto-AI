#!/usr/bin/env python
# coding: utf-8
Step1: importing libraries
# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Step2: Importing dataset
# In[2]:


dataset= pd.read_csv('cancer.csv')


# In[3]:


dataset


# In[4]:


type(dataset)


# In[5]:


dataset.isnull().any()

Step3: Seperating independent and dependent variables
# In[6]:


x= dataset.iloc[:,0:5].values


# In[7]:


x


# In[8]:


y= dataset.iloc[:,5:].values


# In[9]:


y


# In[10]:


plt.scatter(x,y)          #to know if there is linear relation between x and y

Step4: Splitting train and test
# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[ ]:


x_train


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr= LinearRegression()


# In[ ]:


lr.fit(x_train,y_train)


# In[ ]:


x_test


# In[ ]:


lr.predict(x_test)        #it will give the prediction of x test


# In[ ]:


y_test


# In[ ]:


lr.predict([[12.3]])


# In[ ]:


plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,lr.predict(x_train),color="green")


# In[ ]:


plt.scatter(x_test,y_test,color="yellow")
plt.plot(x_test,lr.predict(x_test),color="red")


# In[ ]:


from sklearn.metrics import r2_score             #metrics of regression to find accuracy
r2_score(y_test,lr.predict(x_test))


# In[ ]:




