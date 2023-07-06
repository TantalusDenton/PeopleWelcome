#!/usr/bin/env python
# coding: utf-8

# In[45]:


import logging
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub

from datetime import datetime
from keras.preprocessing import image
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from tensorflow.keras import layers

from utils import *

warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# In[46]:


print("TF version:", tf.__version__)


# In[47]:


movies = pd.read_csv("/Users/avni/Desktop/Movie_Poster/MovieGenre.csv", encoding="ISO-8859-1")
movies.head()


# In[48]:


# Removes rows with missing Id, Genre or Poster 
movies.dropna(subset=['imdbId', 'Genre', 'Poster'], inplace=True)

# Removes "Adult" movies
movies.drop(movies[movies['Genre'].str.contains('Adult')].index, inplace=True)

movies.head(3)


# In[49]:


# Define destination folder
destination = '/Users/avni/Desktop/Movie_Poster/MoviePosters'
# Download in parallel and return the successful subset of the movies dataframe
#movies = download_parallel(movies, destination)


# In[50]:


movies = download_parallel(movies, destination)


# In[51]:


print("Number of movie posters to keep:", len(movies))

# Save the final movies dataframe to disk

munge_dir = "./munge"
if not os.path.exists(munge_dir):
    os.makedirs(munge_dir)
movies.to_csv(os.path.join(munge_dir, "movies.csv"), index=False)


# In[52]:


movies = pd.read_csv("./munge/movies.csv")
print("Number of movie posters in last download: {}\n".format(len(movies)))
movies.head(3)


# In[53]:


label_freq = movies['Genre'].apply(lambda s: str(s).split('|')).explode().value_counts().sort_values(ascending=False)

# create a Bar plot
style.use("fivethirtyeight")
plt.figure(figsize=(12,10))
sns.barplot(y=label_freq.index.values, x=label_freq, order=label_freq.index)
plt.title("Label frequency", fontsize=14)
plt.xlabel("")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[54]:


rare = list(label_freq[label_freq<1000].index)
print("Ignoring these rare labels:", rare)


# In[55]:


# move Genre into a list of labels and remove the other ones
movies['Genre'] = movies['Genre'].apply(lambda s: [l for l in str(s).split('|') if l not in rare])
movies.head()


# In[56]:


X_train, X_val, y_train, y_val = train_test_split(movies['imdbId'], movies['Genre'], test_size=0.2, random_state=44)
print("Number of posters for training: ", len(X_train))
print("Number of posters for validation: ", len(X_val))


# In[57]:


X_train = [os.path.join('/Users/avni/Desktop/Movie_Poster/MoviePosters', str(f)+'.jpg') for f in X_train]
X_val = [os.path.join('/Users/avni/Desktop/Movie_Poster/MoviePosters', str(f)+'.jpg') for f in X_val]
X_train[:3]


# In[58]:


y_train = list(y_train)
y_val = list(y_val)
y_train[:3]


# In[59]:


nobs = 8 # Maximum number of images to display
ncols = 4 # Number of columns in display
nrows = nobs//ncols # Number of rows in display

style.use("default")
plt.figure(figsize=(12,4*nrows))
for i in range(nrows*ncols):
    ax = plt.subplot(nrows, ncols, i+1)
    plt.imshow(Image.open(X_train[i]))
    plt.title(y_train[i], size=10)
    plt.axis('off')


# In[60]:


# Fit the multi-label binarizer on the training set
print("Labels:")
mlb = MultiLabelBinarizer()
mlb.fit(y_train)

# Loop over all labels and show them
N_LABELS = len(mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))


# In[61]:


y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)


# In[62]:


for i in range(3):
    print(X_train[i], y_train_bin[i])


# In[63]:


IMG_SIZE = 224 # Specify height and width of image to match the input format of the model
CHANNELS = 3 


# In[64]:


def parse_function(filename, label):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, label


# In[65]:


BATCH_SIZE = 256 # measure an F1-score
AUTOTUNE = tf.data.experimental.AUTOTUNE # preprocessing
SHUFFLE_BUFFER_SIZE = 1024


# In[66]:


def create_dataset(filenames, labels, is_training=True):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    
    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        
    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset


# In[67]:


train_ds = create_dataset(X_train, y_train_bin)
val_ds = create_dataset(X_val, y_val_bin)


# In[68]:


for f, l in train_ds.take(1):
    print("Shape of features array:", f.numpy().shape)
    print("Shape of labels array:", l.numpy().shape)


# In[69]:


feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))


# In[70]:


feature_extractor_layer.trainable = False


# In[71]:


model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(1024, activation='relu', name='hidden_layer'),
    layers.Dense(N_LABELS, activation='sigmoid', name='output')
])

model.summary()


# In[72]:


for batch in train_ds:
    print(model.predict(batch)[:1])
    break


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




