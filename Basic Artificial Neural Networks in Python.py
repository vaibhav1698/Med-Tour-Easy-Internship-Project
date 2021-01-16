#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


# In[2]:


#Dummy Data for Training
np.random.seed(0)

feature_set, labels = datasets.make_moons(100, noise=0.1)
plt.figure(figsize=(10,7))
plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap=plt.cm.winter)

labels = labels.reshape(100,1)

plt.show;


# In[3]:


np.random.seed(0)

features, labels = datasets.make_moons(100, noise=0.1)
plt.figure(figsize=(10,7))
plt.scatter(features[:,0], features[:,1], c=labels, cmap=plt.cm.winter)

labels = labels.reshape(100,1)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_der(x):
    return sigmoid(x) * (1-sigmoid(x))
wh = np.random.rand(len(features[0]), 4)
wo = np.random.rand(4,1)

lr=0.5

for epoch in range(20000):
    ##Feedforward Propagation
    zh = np.dot(features, wh)
    ah = sigmoid(zh)
    
    zo = np.dot(ah,wo)
    ao = sigmoid(zo)
    
    #Phase 1
    error_out = ((1/2) * (np.power((ao - labels),2)))
    print(error_out.sum())
    dcost_dao = ao - labels
    dao_dzo = sigmoid_der(zo)
    dzo_dwo = ah
    
    dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)
    
    #Phase 2

dcost_dzo = dcost_dao * dao_dzo

dzo_dah = wo
dcost_dah = np.dot(dcost_dzo, dzo_dah.T)

dah_dzh = sigmoid_der(zh)
dzh_dwh = feature_set
dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)


#Update Weights

wh -= lr * dcost_wh
wo -= lr * dcost_wo


# In[ ]:


import tensorflow as tf
import numpy as np

features = tf.keras.Input(shape=(4,))
hidden = tf.keras.layers.Dense(3, activation=tf.nn.sigmoid)(features)
labels = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden)
model = tf.keras.Model(inputs=inputs, outputs=labels)

mse = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.sgd(learning_rate=1e-1)

np.random.seed(0)

features, labels = datasets.make_moons(200, noise=0.10)
plt.figure(figsize=(10,7))
plt.scatter(features[:, 0], features[:,1], c=labels, cmap=plt.cm.winter)

for _ in range(20000):
    for step, (x,y) in enumerate(zip(x_data, y_data)):
        model.train_on_batch(np.array([x]), np.array([y]))
        
x = tf.convert_to_tensor(np.array(x_data[0]), dtype=tf.float64)
print(model(x).numpy())


# In[ ]:




