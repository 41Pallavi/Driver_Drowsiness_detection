#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil 
from tqdm import tqdm


# #subject ID:
# xxx
# 
# image number:
# xxx
# 
# gender:
# 0 - male
# 1 - famale
# 
# glasses:
# 0 - no
# 1 - yes
# 
# eye state:
# 0 - close
# 1 - open
# 
# reflections:
# 0 - none
# 1 - low
# 2 - high
# 
# lighting conditions/image quality:
# 0 - bad
# 1 - good
# 
# sensor type:
# 01 - RealSense SR300 640x480
# 02 - IDS Imaging, 1280x1024
# 03 - Aptina Imagin 752x480
# 
# example:
# s001_00123_0_0_0_0_0_01.png

# In[2]:


Raw_DIR =r'C:\Users\91993\Desktop\driver_drowsiness_detection\mrlEyes_2018_01'
for dirpath, dirname, filenames in os.walk(Raw_DIR):
     for i in tqdm([f for f in filenames if f.endswith('.png')]):
         if i.split('_')[4] == '0':
                shutil.copy(src = dirpath + '/' + i, dst = r'C:\Users\91993\Desktop\driver_drowsiness_detection\Prepared_data\closedEye')
         elif i.split('_')[4] == '1':
                shutil.copy(src=dirpath + '/' + i, dst = r'C:\Users\91993\Desktop\driver_drowsiness_detection\Prepared_data\openEye')


# In[3]:


get_ipython().system('pip install tensorflow==2.7.0')


# In[4]:


pip install protobuf==3.20.0


# In[5]:


import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data Augumentation


# In[6]:


train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 0.2,shear_range = 0.2,
    zoom_range = 0.2,width_shift_range = 0.2,
    height_shift_range = 0.2, validation_split = 0.2)

train_data = train_datagen.flow_from_directory(
    r'C:\Users\91993\Desktop\driver_drowsiness_detection\Prepared_data\train',
    target_size=(80, 80),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)


validation_data = train_datagen.flow_from_directory(
    r'C:\Users\91993\Desktop\driver_drowsiness_detection\Prepared_data\train',
    target_size=(80, 80),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)


# In[7]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_data = test_datagen.flow_from_directory(
    r'C:\Users\91993\Desktop\driver_drowsiness_detection\Prepared_data\test',
    target_size=(80, 80),
    batch_size=8,
    class_mode='categorical'
)


# In[ ]:




