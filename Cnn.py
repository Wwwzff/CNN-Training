#!/usr/bin/env python
# coding: utf-8

# In[3]:


from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm_notebook
from random import shuffle
import shutil
import pandas as pd


# In[12]:


def organize_datasets(path_to_data,n,ratio):
    files = os.listdir(path_to_data)
    #print(files)
    files = [os.path.join(path_to_data,f) for f in files]
    shuffle(files)
    files = files[:n]
    
    n = int(len(files)*ratio)
    val,train = files[:n],files[n:]
    
    shutil.rmtree('/home/chester/Downloads/all/data')
    print('/data/ removed')
    
    for c in ['dogs','cats']:
        os.makedirs('/home/chester/Downloads/all/data/train/{0}/'.format(c))
        os.makedirs('/home/chester/Downloads/all/data/validation/{0}/'.format(c))
    print('folder created!')
    
    for t in tqdm_notebook(train):
        if 'cat' in t:

            shutil.copy2(t,'/home/chester/Downloads/all/data/train/cats')
        else:
            shutil.copy2(t, '/home/chester/Downloads/all/data/train/dogs')
                         
    for v in tqdm_notebook(val):
        if 'cat' in v:
            shutil.copy2(v,'/home/chester/Downloads/all/data/validation/cats')
        else:
            shutil.copy2(v,'/home/chester/Downloads/all/data/validation/dogs')
            
    print('Data copied!')

n = 25000
ratio = 0.2
organize_datasets(path_to_data='/home/chester/Downloads/all/training/',n=n,ratio=ratio)


# In[13]:


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras_tqdm import TQDMNotebookCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback


# In[14]:


batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1/255.,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)
val_datagen = ImageDataGenerator(rescale = 1/255.)


# In[15]:


train_generator = train_datagen.flow_from_directory('/home/chester/Downloads/all/data/train/',
                                                   target_size = (225,225),
                                                   batch_size = batch_size,
                                                   class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory('/home/chester/Downloads/all/data/validation/',
                                                      target_size = (225,225),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')


# In[20]:


model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = (225,225,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

epochs = 50
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.summary()


# In[21]:


class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self,batch,logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

history = LossHistory()

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=0,
                                              patience=2,
                                              verbose=0,
                                              mode='auto')


# In[22]:


fitted_model = model.fit_generator(train_generator,
                                   steps_per_epoch=int(n*(1-ratio))//batch_size,
                                   epochs = epochs,
                                   validation_data=validation_generator,
                                   validation_steps=int(n*ratio)//batch_size,
                                   callbacks = [TQDMNotebookCallback(leave_inner=True,leave_outer=True),early_stopping,history],
                                   verbose=0
                                  )


# In[23]:


model.save('/home/chester/Downloads/all/model.h5')


# In[24]:


losses,val_losses = history.losses, history.val_losses
fig = plt.figure(figsize = (15,5))
plt.plot(fitted_model.history['loss'],'g',label='train losses')
plt.plot(fitted_model.history['val_loss'],'r',label='validation losses')
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

