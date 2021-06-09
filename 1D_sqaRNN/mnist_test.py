#!/usr/bin/env python
# coding: utf-8

# In[27]:


import os, sys


# In[26]:


os.environ['LD_LIBRARY_PATH']


# In[28]:


sys.version


# In[1]:


import tensorflow as tf


# In[2]:


tf.test.gpu_device_name()


# In[3]:


tf.test.is_gpu_available()


# In[4]:


tf.config.list_physical_devices('GPU')


# In[16]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# In[17]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])


# In[18]:


predictions = model(x_train[:1]).numpy()
predictions


# In[19]:


tf.nn.softmax(predictions).numpy()


# In[20]:


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# In[21]:


loss_fn(y_train[:1], predictions).numpy()


# In[22]:


model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


# In[23]:


model.fit(x_train, y_train, epochs=5000)


# In[13]:


model.evaluate(x_test,  y_test, verbose=2)


# In[14]:


probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])


# In[15]:


probability_model(x_test[:5])


# In[ ]:




