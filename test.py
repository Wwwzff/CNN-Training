#!/usr/bin/env python
# coding: utf-8

# In[6]:


from keras.models import load_model
from PIL import Image 
import matplotlib.pyplot as plt  
import numpy as np
import os 
import tensorflow as tf

def predict(model_path,image_path):
    model = load_model(model_path)
    img = Image.open(image_path)
    plt.imshow(img)
    img = img.resize(size = [225,225])
    a = np.array(img).reshape(1,225,225,3)
    score = model.predict(a)
    if(str(score)=='[[0. 1.]]'):
        print("it's a dog")
    else:
        print("it's a cat")

model_path = '/home/chester/Downloads/all/model.h5'
image_path = '/home/chester/Documents/dog/cat.jpg'
predict(model_path,image_path)


# In[ ]:




