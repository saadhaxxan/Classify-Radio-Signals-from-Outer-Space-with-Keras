#!/usr/bin/env python
# coding: utf-8

# <h2 align=center> Classify Radio Signals from Outer Space with Keras</h2>

# ![](Allen_Telescope.jpg)
# [Allen Telescope Array](https://flickr.com/photos/93452909@N00/5656086917) by [brewbooks](https://www.flickr.com/people/93452909@N00) is licensed under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)

# ##  Import Libraries

# In[2]:


from livelossplot.tf_keras import PlotLossesCallback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn import metrics

import numpy as np
np.random.seed(42)
import warnings;warnings.simplefilter('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
print('Tensorflow version:', tf.__version__)


# ##  Load and Preprocess SETI Data

# In[10]:


train_images = pd.read_csv('dataset/train/images.csv',header=None)
train_labels = pd.read_csv('dataset/train/labels.csv',header=None)
validation_images = pd.read_csv('dataset/validation/images.csv',header=None)
validation_labels = pd.read_csv('dataset/validation/labels.csv',header=None)


# In[11]:


train_images.head()


# In[12]:


train_labels.head()
#0 == squigle
#1 == narrow band
#2 == nOise
#3 == narrow band drd


# In[13]:


train_images.shape


# In[14]:


train_labels.shape


# In[15]:


validation_images.shape


# In[16]:


validation_labels.shape


# In[17]:


x_train = train_images.values.reshape(3200,64,128,1)


# In[18]:


x_val = validation_images.values.reshape(800,64,128,1)


# In[19]:


y_train = train_labels.values
y_val = validation_labels.values


# ## 2D Spectrograms

# In[20]:


plt.figure(0,figsize=(12,12))
for i in range(1,10):
    plt.subplot(3,3,i)
    image = np.squeeze(x_train[np.random.randint(0,3200)])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image,cmap='gray')


# In[21]:


plt.figure(0,figsize=(12,12))
for i in range(1,4):
    plt.subplot(1,3,i)
    image = np.squeeze(x_train[np.random.randint(0,3200)])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image,cmap='jet')


# ##  Create Training and Validation Data Generators

# In[22]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(horizontal_flip=True)


# In[23]:


datagen_train.fit(x_train)
datagen_val = ImageDataGenerator(horizontal_flip=True)
datagen_val.fit(x_val)


# ##  Creating the CNN Model

# In[24]:


from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint


# In[28]:


# Initialising the CNN
model = Sequential()
# 1st Convolution
model.add(Conv2D(32,(5,5),padding='same',input_shape=(64,128,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# 2nd Convolution layer
model.add(Conv2D(64,(5,5),padding='same',input_shape=(64,128,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# Flattening
model.add(Flatten())
# Fully connected layer
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
# final prediction layer
model.add(Dense(4,activation='softmax'))


# ##  Learning Rate Scheduling and Compile the Model

# In[33]:


initial_learning  = 0.005
lr_schedule =  tf.keras.optimizers.schedules.ExponentialDecay( initial_learning_rate = initial_learning,decay_rate = 0.96,decay_steps = 5, staircase = True)


# In[34]:


optimizer = Adam(learning_rate = lr_schedule)


# In[49]:


model.compile(optimizer= optimizer,metrics=['accuracy'],loss='categorical_crossentropy')


# ##  Training the Model

# In[47]:


model.summary()


# In[37]:


checkpoint = ModelCheckpoint('weights.h5',monitor = 'val_loss',save_weights_only = True, mode= 'min', verbose = 0)


# In[38]:


callbacks = [PlotLossesCallback,checkpoint]


# In[39]:


batch_size = 32


# In[50]:


history = model.fit_generator(datagen_train.flow(x_train,y_train,shuffle=True,batch_size=batch_size),
                   steps_per_epoch = len(x_train)//batch_size,
                   validation_data= datagen_val.flow(x_val,y_val,shuffle=True,batch_size=batch_size),
                   validation_steps = len(x_val)//batch_size,
                   epochs=10
                   callbacks=[checkpoint]
                   )


# ## Task 8: Model Evaluation

# In[51]:


model.evaluate(x_val,y_val)


# In[52]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns

y_true = np.argmax(y_val,1)
y_pred = np.argmax(model.predict(x_val),1)


# In[54]:


print(metrics.classification_report(y_true,y_pred))


# In[55]:


print(metrics.accuracy_score(y_true,y_pred))


# In[ ]:


labels = ["squiggle", "narrowband", "noise", "narrowbanddrd"]

