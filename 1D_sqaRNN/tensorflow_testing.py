#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import tensorflow as tf

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config....)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

tf.compat.v1.disable_eager_execution()
    
hello = tf.constant('Hello,TensorFlow')
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.compat.v1.Session(config=config)
print(sess.run(hello))


# In[2]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

input = tf.Variable(tf.random.normal([100, 28, 28, 1]))
filter = tf.Variable(tf.random.normal([5, 5, 1, 6]))


sess = tf.compat.v1.Session()
#tf.compat.v1.disable_eager_execution()
sess.run(tf.compat.v1.global_variables_initializer())



op = tf.nn.conv2d(input, filter, strides = [1, 1, 1, 1], padding = 'VALID')
out = sess.run(op)


# In[ ]:




