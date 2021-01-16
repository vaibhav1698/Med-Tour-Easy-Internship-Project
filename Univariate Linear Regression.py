#!/usr/bin/env python
# coding: utf-8

# <img src="https://rhyme.com/assets/img/logo-dark.png" align="center">

# <h2 align="center"> Univariate Linear Regression </h2>

#  

# ### Task 2: Load the Data and Libraries
# ---

# In[1]:


import matplotlib.pyplot as plt 
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import numpy as np
import pandas as pd  
import seaborn as sns 
plt.rcParams['figure.figsize'] = (12, 8)


# In[3]:


data = pd.read_csv('bike_sharing_data.txt')
data.head()


# In[4]:


data.info()
print('\n')
data.describe()
print('\n')
data.shape


#  

# ### Task 3: Visualize the Data
# ---

# In[5]:


ax = sns.scatterplot(x="Population", y="Profit", data=data)
ax.set_title("Profit in $10000s vs City Population in 10000s");


#  

#  

# ### Task 4: Compute the Cost $J(\theta)$
# ---

# The objective of linear regression is to minimize the cost function
# 
# $$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)} )^2$$
# 
# where $h_{\theta}(x)$ is the hypothesis and given by the linear model
# 
# $$h_{\theta}(x) = \theta^Tx = \theta_0 + \theta_1x_1$$

# In[6]:


def cost_function(x, y, theta):
    m = len(y)
    y_pred = x.dot(theta)
    error = (y_pred - y) ** 2
    
    return (1/(2*m)) * np.sum(error)


# In[7]:


m = data.Population.values.size
x = np.append(np.ones((m, 1)), data.Population.values.reshape(m, 1), axis=1)
y = data.Profit.values.reshape(m, 1)
theta = np.zeros((2,1))

cost_function(x, y, theta)


#  

# ### Task 5: Gradient Descent
# ---

# Minimize the cost function $J(\theta)$ by updating the below equation and repeat unitil convergence
#         
# $\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})x_j^{(i)}$ (simultaneously update $\theta_j$ for all $j$).

# In[8]:


def gradient_descent(x, y, theta, alpha, iterations):
    m = len(y)
    costs = []
    for i in range(iterations):
        y_pred = x.dot(theta)
        error = np.dot(x.transpose(), (y_pred - y))
        theta -= alpha * 1/m * error
        costs.append(cost_function(x, y, theta))
    return theta, costs


# In[9]:


theta, costs = gradient_descent(x, y, theta, alpha=0.01, iterations=2000)

print("h(x) = {} + {}x1".format(str(round(theta[0, 0], 2)),
                               str(round(theta[1, 0],2))))


# In[ ]:





# ### Task 6: Visualising the Cost Function $J(\theta)$
# ---

# In[10]:


from mpl_toolkits.mplot3d import Axes3D


# In[11]:


theta_0 = np.linspace(-10, 10, 100)
theta_1 = np.linspace(-1, 1, 100)

cost_values = np.zeros((len(theta_0), len(theta_1)))

for i in range(len(theta_0)):
    for j in range(len(theta_1)):
        t = np.array(theta_0[i], theta_1[j])
        cost_values[i, j] = cost_function(x, y, t)


# In[12]:


fig = plt.figure(figsize= (12,8))
ax = fig.gca(projection = '3d')

surf = ax.plot_surface(theta_0, theta_1, cost_values, cmap = 'viridis')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.xlabel("$\Theta_0$")
plt.ylabel("$\Theta_1$")
ax.set_zlabel("$J(\Theta)$")
ax.view_init(30, 330)

plt.show()


#  

# ### Task 7: Plotting the Convergence
# ---

# Plot $J(\theta)$ against the number of iterations of gradient descent:

# In[13]:


plt.plot(costs)
plt.xlabel("iterations")
plt.ylabel("$J(\Theta)$")
plt.title("Value of the cost function over Iterations of Gradient Descent")


#  

# ### Task 8: Training Data with Linear Regression Fit
# ---

# In[14]:


theta.shape


# In[15]:


theta = np.squeeze(theta)
sns.scatterplot(x="Population", y="Profit", data=data)

x_value = [x for x in range(5, 25)]
y_value = [(x * theta[1] + theta[0]) for x in x_value]
sns.lineplot(x_value, y_value)

plt.xlabel("Population in 10000s")
plt.ylabel("Profit in $10,000s")
plt.title("Linear Regressio Fit")
plt.show()


# In[ ]:





# ### Task 9: Inference using the optimized $\theta$ values
# ---

# $h_\theta(x) = \theta^Tx$

# In[16]:


def predict(x, theta):
    y_pred = np.dot(theta.transpose(), x)
    return y_pred


# In[17]:


y_pred_1 = predict(np.array([1,4]), theta) * 10000
print("For a population of 40,000 people, the model predicts a profit of $" + str(round(y_pred_1, 0)))


# In[ ]:




