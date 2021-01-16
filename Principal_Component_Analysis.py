#!/usr/bin/env python
# coding: utf-8

# <h2 align=center> Principal Component Analysis</h2>

#  

# ### Task 2: Load the Data and Libraries
# ---

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[5]:


plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12,8)


# In[6]:


# data URL: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

iris = pd.read_csv("iris.data")
iris.head()


# In[7]:


iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris.dropna(how='all', inplace=True)
iris.head()


# In[8]:


iris.info()
print('\n')
iris.describe()


#  

# ### Task 3: Visualize the Data
# ---

# In[9]:


sns.scatterplot(x = iris.sepal_length, y = iris.sepal_width,
               hue = iris.species, style = iris.species)


#  

# ### Task 4: Standardize the Data
# ---

# In[10]:


x = iris.iloc[:, 0:4].values #taking all rows and columns 0 to 4 excluding 4
y = iris.species.values


# In[11]:


from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(x) #subtraccting mean from each and then dividing by standard deviation


#  

# ### Task 5: Compute the Eigenvectors and Eigenvalues
# ---

# Covariance: $\sigma_{jk} = \frac{1}{n-1}\sum_{i=1}^{N}(x_{ij}-\bar{x_j})(x_{ik}-\bar{x_k})$
# 
# Coviance matrix: $Σ = \frac{1}{n-1}((X-\bar{x})^T(X-\bar{x}))$

# In[12]:


covariance_matrix = np.cov(X.T)
print("Covariance matrix: \n", covariance_matrix)


# We can prove this by looking at the covariance matrix. It has the property that it is symmetric. We also constrain the each of the columns (eigenvectors) such that the values sum to one. Thus, they are orthonormal to each other.
# 
# Eigendecomposition of the covriance matrix:  $Σ = W\wedge W^{-1}$

# In[13]:


eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
print("Eigen Vectors: \n", eigen_vectors, "\n")
print("Eigen Vectors: \n", eigen_values)


#  

# ### Task 6: Singular Value Decomposition (SVD)
# ---

# In[14]:


eigen_vector_svd, s, v = np.linalg.svd(X.T)

eigen_vector_svd


# In[ ]:





#  

# ### Task 7: Picking Principal Components Using the Explained Variance
# ---

# In[15]:


for val in eigen_values:
    print(val)


# In[16]:


variance_explained = [(i/sum(eigen_values))*100 for i in eigen_values]

variance_explained
#output represents percentage of variance explained by each component
#72.7% variance of our data is explained by 1st principal component(example)


# In[17]:


cumulative_variance_explained = np.cumsum(variance_explained)
cumulative_variance_explained


# In[18]:


sns.lineplot(x = [1,2,3,4], y = cumulative_variance_explained);
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Explained variance vs Number of components")
plt.show()


#  

# ### Task 8: Project Data Onto Lower-Dimensional Linear Subspace
# ---

# In[19]:


eigen_vectors
#we are only using first two principal component since they explain 95% of the variance
#first two principal components are the first two columns of the eigen_vectors
#explaining variance means that data will give more accurate prediction


# In[20]:


projection_matrix = (eigen_vectors.T[:][:])[:2].T
print("Projection matrix: \n", projection_matrix)


# In[21]:


x_pca = x.dot(projection_matrix)


# In[24]:


for species in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
    sns.scatterplot(x_pca[y==species, 0],
                    x_pca[y==species, 1])


# In[ ]:




