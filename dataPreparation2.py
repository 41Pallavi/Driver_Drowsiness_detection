#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(directory, img_size=100, augment=True):
    """
    Function to preprocess image data for a classification model.

    Parameters:
    - directory: Path to the directory containing the dataset folders.
    - img_size: Desired size for the images after resizing. Default is 100.
    - augment: Boolean flag indicating whether to apply data augmentation. Default is False.

    Returns:
    - X: Preprocessed image data as a numpy array.
    - y: Labels for the image data.
    """
    # Define labels and initialize empty lists to store data
    categories = ['Closed', 'Open']  # Adjust labels based on folder names
    X = []
    y = []

    # Initialize ImageDataGenerator for data augmentation
    if augment:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

    # Loop through each category
    for category in categories:
        path = os.path.join(directory, category)
        label = categories.index(category)
        # Loop through each image in the category folder
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            # Read the original image
            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            # Resize the image
            img = cv2.resize(original_img, (img_size, img_size))
            # If augment flag is True, apply data augmentation
            if augment:
                img = datagen.random_transform(img)
            # Append the preprocessed image and its label to the lists
            X.append(img)
            y.append(label)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y


# In[5]:


# Path to the directory containing the dataset folders
dataset_dir = r"C:\Users\91993\Desktop\driver_drowsiness_detection\dataset2\train"

# Preprocess the data
X, y = preprocess_data(dataset_dir)


# In[7]:


import random

# Get the number of images in the dataset
num_images = len(X)

# Generate a random index within the range of available images
random_index = random.randint(0, num_images - 1)

# Get the random image and its label
random_img = X[random_index]
random_label = y[random_index]

# Display the original and preprocessed images

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(random_img)
axes[0].set_title('Original Image (Label: {})'.format(random_label))
axes[0].axis('off')

# Preprocess the image by converting it to grayscale and resizing
preprocessed_img = cv2.cvtColor(random_img, cv2.COLOR_RGB2GRAY)
preprocessed_img = cv2.resize(preprocessed_img, (100, 100))

axes[1].imshow(preprocessed_img, cmap='gray')
axes[1].set_title('Preprocessed Image')
axes[1].axis('off')

plt.show()







# In[ ]:




