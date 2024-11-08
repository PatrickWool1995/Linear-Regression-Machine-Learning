#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv(r"PLACEHOLDER.csv")


# In[6]:


#find the mean total production of honey per year

ppy = df.groupby('year').totalprod.mean().reset_index()
display(ppy)


# In[14]:


#create X value that is year column in ppy and y value that is total production

x = ppy['year']
x = x.values.reshape(-1,1)

y = ppy['totalprod']
y = y.values.reshape(-1,1)


# In[22]:


#plot values
plt.scatter(x,y)

#create linear regression model
lr = linear_model.LinearRegression()
lr.fit(x,y)
y_predict = lr.predict(x)

#plot line
plt.plot(x,y_predict)
plt.show()


# In[27]:


#predict and plot honey production from 2013 to 2024

X_future = np.array(range(2013,2025))
X_future = X_future.reshape(-1,1)

future_predict = lr.predict(X_future)

plt.plot(X_future,future_predict)
plt.show()


# In[ ]:





# In[ ]:




