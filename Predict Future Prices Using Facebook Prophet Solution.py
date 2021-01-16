#!/usr/bin/env python
# coding: utf-8

# # PREDICTING FUTURE PRODUCT PRICES USING FACEBOOK PROPHET 
# 
# 

# # TASK #1: PROJECT OVERIEW 

# 
# ![image.png](attachment:image.png)
# 

# 

# # TASK #2: IMPORT LIBRARIES AND DATASET

# - You must install fbprophet package as follows: 
#      pip install fbprophet
#      
# - If you encounter an error, try: 
#     conda install -c conda-forge fbprophet
# 
# 

# In[1]:


# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import random
import seaborn as sns
from fbprophet import Prophet


# In[2]:


# dataframes creation for both training and testing datasets 
avocado_df = pd.read_csv('avocado.csv')


# 
# - Date: The date of the observation
# - AveragePrice: the average price of a single avocado
# - type: conventional or organic
# - year: the year
# - Region: the city or region of the observation
# - Total Volume: Total number of avocados sold
# - 4046: Total number of avocados with PLU 4046 sold
# - 4225: Total number of avocados with PLU 4225 sold
# - 4770: Total number of avocados with PLU 4770 sold

# In[3]:


# Let's view the head of the training dataset
avocado_df.head()


# In[4]:


# Let's view the last elements in the training dataset
avocado_df.tail(20)


# In[5]:


avocado_df.describe()


# In[6]:


avocado_df.info()


# In[7]:


avocado_df.isnull().sum()


# # TASK #3: EXPLORE DATASET  

# In[8]:


avocado_df = avocado_df.sort_values("Date")


# In[9]:


plt.figure(figsize=(10,10))
plt.plot(avocado_df['Date'], avocado_df['AveragePrice'])


# In[10]:


plt.figure(figsize=(10,6))
sns.distplot(avocado_df["AveragePrice"], color = 'b')


# In[11]:


avocado_df


# In[12]:


sns.violinplot(y="AveragePrice", x="type", data = avocado_df)


# In[13]:


# Bar Chart to indicate the number of regions 

sns.set(font_scale=0.7) 
plt.figure(figsize=[25,12])
sns.countplot(x = 'region', data = avocado_df)
plt.xticks(rotation = 45)


# In[14]:


# Bar Chart to indicate the count in every year
sns.set(font_scale=1.5) 
plt.figure(figsize=[25,12])
sns.countplot(x = 'year', data = avocado_df)
plt.xticks(rotation = 45)


# In[15]:


conventional = sns.catplot('AveragePrice','region', data = avocado_df[ avocado_df['type']=='conventional'],
                  hue='year',
                  height=20)


# In[16]:


organic = sns.catplot('AveragePrice','region', data = avocado_df[ avocado_df['type']=='organic'],
                  hue='year',
                  height=20)


# # TASK 4: PREPARE THE DATA BEFORE APPLYING FACEBOOK PROPHET TOOL 

# In[17]:


avocado_prophet_df = avocado_df[['Date', 'AveragePrice']] 


# In[18]:


avocado_prophet_df


# In[19]:


avocado_prophet_df = avocado_prophet_df.rename(columns={'Date':'ds', 'AveragePrice':'y'})


# In[20]:


avocado_prophet_df


# # TASK 5: UNDERSTAND INTUITION BEHIND FACEBOOK PROPHET

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # TASK 6: DEVELOP MODEL AND MAKE PREDICTIONS - PART A

# In[21]:


m = Prophet()
m.fit(avocado_prophet_df)


# In[22]:


# Forcasting into the future
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[23]:


forecast


# In[24]:


figure = m.plot(forecast, xlabel='Date', ylabel='Price')


# In[25]:


figure3 = m.plot_components(forecast)


# # TASK 7: DEVELOP MODEL AND MAKE PREDICTIONS (REGION SPECIFIC) - PART B

# In[26]:


# dataframes creation for both training and testing datasets 
avocado_df = pd.read_csv('avocado.csv')


# In[27]:


avocado_df


# In[28]:


avocado_df_sample = avocado_df[avocado_df['region']=='West']


# In[29]:


avocado_df_sample


# In[30]:


avocado_df_sample


# In[31]:


avocado_df_sample = avocado_df_sample.sort_values("Date")


# In[32]:


avocado_df_sample


# In[33]:


plt.figure(figsize=(10,10))
plt.plot(avocado_df_sample['Date'], avocado_df_sample['AveragePrice'])
plt.xlabel('Price')


# In[34]:


avocado_df_sample = avocado_df_sample.rename(columns={'Date':'ds', 'AveragePrice':'y'})


# In[35]:


m = Prophet()
m.fit(avocado_df_sample)
# Forcasting into the future
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[36]:


figure = m.plot(forecast, xlabel='Date', ylabel='Price')


# In[37]:


figure3 = m.plot_components(forecast)


# # GREAT JOB!
