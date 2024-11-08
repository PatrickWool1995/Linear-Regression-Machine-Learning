#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"PLACEHOLDER.csv")

display(df.head())


# In[10]:


#Set up X variable with columns that will be tested to see if there is correlation with a higher winnings

x = df[['FirstServePointsWon','BreakPointsFaced','Aces','ServiceGamesWon','TotalServicePointsWon']]
y = df['Winnings']


# In[11]:


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size = 0.8,test_size = 0.2,random_state = 8)

l = LinearRegression()
model = l.fit(x_train,y_train)
print(l.coef_)
print(l.score(x_train,y_train))
print(l.score(x_test,y_test))
y_predict = model.predict(x_test)


# In[12]:


plt.scatter(y_predict,y_test, alpha = 0.4)
plt.show()


# In[ ]:




