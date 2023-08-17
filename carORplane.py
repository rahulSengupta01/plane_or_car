#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import HTML

HTML('<i class="fa fa-check-circle" style="font-size:24px;color:green;"></i> Success!')


# In[2]:


import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img


# In[3]:


from tensorflow.keras.preprocessing.image import load_img,img_to_array
from PIL import Image
img_width=224
img_height=224
img1 = load_img(r"C:\Users\RAHUL\OneDrive\Desktop\Partic\photo-1616455579100-2ceaa4eb2d37.jpg")
img2 = load_img(r"C:\Users\RAHUL\Downloads\v_data\v_data\train\planes\67.jpg")


# In[4]:


img1


# In[5]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense


# In[6]:


img_width=224
img_height=224
train_data=r"C:\Users\RAHUL\Downloads\v_data\v_data\train"
test_data=r"C:\Users\RAHUL\Downloads\v_data\v_data\test"
train_samples=400
test_samples=100
input_shape=(img_width,img_height,3)


# In[7]:


#model define
model=Sequential()
model.add(Conv2D(32,(2,2),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(2,2),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(2,2),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[8]:


model.summary()


# In[9]:


model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


# In[10]:


train_data=ImageDataGenerator(rescale=1./255)
test_data=ImageDataGenerator(rescale=1./255)
train_data_load=train_data.flow_from_directory(
    "C:/Users/RAHUL/Downloads/v_data/v_data/train",
    target_size=(img_width,img_height),
    batch_size=20,
    class_mode='binary')
test_data_load=test_data.flow_from_directory(
    "C:/Users/RAHUL/Downloads/v_data/v_data/test",
    target_size=(img_width,img_height),
    batch_size=20,
    class_mode='binary')


# In[ ]:


model.fit_generator(
    train_data_load,
    epochs=5,
    validation_data=test_data_load,)


# In[11]:


i=img_to_array(img1)
img1


# In[12]:


i.shape
i=Image.fromarray(i.astype('uint8'))
i=i.resize((224,224))
i=img_to_array(i)


# In[13]:


i=i.reshape(1,224,224,3)


# In[14]:


i


# In[15]:


model.predict(i)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




