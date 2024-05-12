#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow==2.7.0')


# In[2]:


pip install protobuf==3.20.0


# In[3]:


import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data Augumentation


# In[4]:


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


# In[5]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_data = test_datagen.flow_from_directory(
    r'C:\Users\91993\Desktop\driver_drowsiness_detection\Prepared_data\test',
    target_size=(80, 80),
    batch_size=8,
    class_mode='categorical'
)


# In[6]:


from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.layers import Dropout,Input,Flatten,Dense,MaxPooling2D


# In[7]:


tf.test.is_gpu_available()


# In[8]:


batchsize=8


# In[9]:


#base model
# in input_tensor if 1 means grey and if colored its 3
bmodel = InceptionV3(include_top=False, weights='imagenet', input_tensor=Input(shape=(80,80,3)))
hmodel = bmodel.output
hmodel = Flatten()(hmodel)
hmodel = Dense(64, activation='relu')(hmodel)
hmodel = Dropout(0.5)(hmodel) #to prevent overfitting
hmodel = Dense(2,activation= 'softmax')(hmodel) #softmax because of categorical
model = Model(inputs=bmodel.input, outputs= hmodel)

#we do not need to train from scratch we only have to train the last layer not use to train the base model
for layer in bmodel.layers:
    layer.trainable = False


# In[10]:


model.summary()


# In[11]:


from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau


# In[ ]:


#verbose is for progress
checkpoint = ModelCheckpoint(r'C:\Users\91993\Desktop\driver_drowsiness_detection\models',monitor='val_loss',save_best_only=True,verbose=3)

#for seven epoches its not incresing then stop 
earlystop = EarlyStopping(monitor = 'val_loss', patience=5, verbose= 3, restore_best_weights=True)

#with how much speed it is learning
learning_rate = ReduceLROnPlateau(monitor= 'val_loss', patience=3, verbose= 3, )

callbacks=[checkpoint,earlystop,learning_rate]


# In[13]:


model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_data,steps_per_epoch=train_data.samples//batchsize,
                   validation_data=validation_data,
                   validation_steps=validation_data.samples//batchsize,
                   callbacks=callbacks,
                    epochs=5)


# In[ ]:


# Model Evaluation


# In[14]:


# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate_generator(test_data, steps=8)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# In[15]:


# Evaluate the model on the test data
train_loss, train_accuracy = model.evaluate_generator(train_data, steps=8)

print("Train Loss:", train_loss)
print("Train Accuracy:", train_accuracy)


# In[16]:


# Evaluate the model on the test data
validation_loss,validation_accuracy = model.evaluate_generator(validation_data, steps=10)

print("validation_Loss:", validation_loss)
print("validation_Accuracy:", validation_accuracy)


# In[17]:


from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


# In[20]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate predictions for the test data
predictions = model.predict_generator(test_data, steps=len(test_data), verbose=1)
y_true = test_data.classes
y_pred = np.argmax(predictions, axis=1)

# Calculate Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Define class labels
class_labels = test_data.class_indices.keys()

# Define labels for axis
labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.2)  # for label size
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)  # font size
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# In[19]:


# Calculate F1 Score
f1 = f1_score(y_true, y_pred)
print("F1 Score:", f1)

# Calculate Precision
precision = precision_score(y_true, y_pred)
print("Precision:", precision)

# Calculate Recall
recall = recall_score(y_true, y_pred)
print("Recall:", recall)


# In[ ]:




