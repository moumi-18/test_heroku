#!/usr/bin/env python
# coding: utf-8

# ### Steps to deploy a ML model
# 1. Train the model
# 2. save as file_name.py

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# In[2]:


dataset = pd.read_csv(r'C:\Users\RONALD\Desktop\IMS-Classroom\Python Code\Deployment\ML\hiring.csv')
dataset.head()


# In[3]:


dataset.isnull().sum()


# In[4]:


#fill null values
dataset['experience'].fillna(0, inplace=True)
dataset['test_score'].fillna(dataset['test_score'].median(), inplace=True)


# In[5]:


dataset.isnull().sum()


# In[6]:


X = dataset.iloc[:,0:3]
X.head()


# In[7]:


y = dataset.iloc[:,-1]
y.head()


# In[8]:


def convert_to_num(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'zero':0, 0:0}
    return word_dict[word]


# In[9]:


X['experience'] = X['experience'].apply(lambda x : convert_to_num(x))


# In[10]:


X.head()


# In[11]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[12]:


regressor.fit(X, y)


# In[13]:


pickle.dump(regressor, open('model.pkl', 'wb'))


# In[14]:


model = pickle.load(open('model.pkl', 'rb'))


# In[15]:


print(model.predict([[2, 9, 6]]))


# #### step 2: make app.py file

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




