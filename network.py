import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


data_path = 'data/processed_PA'

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

# List all Tensorflow module access
print(f"\n##################################################\n TensorFlow has access to the following devices:\n {tf.config.list_physical_devices()} \n##################################################\n")

# Load the dataset
data = tf.keras.utils.image_dataset_from_directory(
    data_path,  # Adjust this path to where your image data is stored
    labels='inferred',
    label_mode='int',
    image_size=(400, 1000),
    batch_size=32
)

# Get class names
class_names = data.class_names

# Create an iterator to retrieve batches
data_iterator = data.as_numpy_iterator()

# Retrieve the first batch of images
batch = data_iterator.next()

# Setup Matplotlib figure and axes
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))

# Display the first four images in the batch
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(np.uint8))  # Convert to uint8 for proper display
    label_index = batch[1][idx]
    ax[idx].title.set_text(class_names[label_index])

plt.show()  # Show the plot

# Scale data
data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()

# Split data
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

train

model = Sequential()

model.add(Conv2D(32, (3,3), 1, activation='relu', input_shape=(400,1000,3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])