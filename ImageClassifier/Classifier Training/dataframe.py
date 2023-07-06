#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print(tf.__version__)


# In[5]:


from sklearn.datasets 
import make_multilabel_classification

# dataset
X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
# dataset shape
print(X.shape, y.shape)
# first few examples
for i in range(10):
 print(X[i], y[i])


# In[7]:


def get_model(n_inputs, n_outputs):
 model = Sequential()
 model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
 model.add(Dense(n_outputs, activation='sigmoid'))
 model.compile(loss='binary_crossentropy', optimizer='adam')
 return model


# In[12]:


#multi-label classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
 
# dataset
def get_dataset():
 X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
 return X, y
 
# model
def get_model(n_inputs, n_outputs):
 model = Sequential()
 model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
 model.add(Dense(n_outputs, activation='sigmoid'))
 model.compile(loss='binary_crossentropy', optimizer='adam')
 return model
 
# repeated k-fold cross-validation
def evaluate_model(X, y):
 results = list()
 n_inputs, n_outputs = X.shape[1], y.shape[1]

 cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

 for train_ix, test_ix in cv.split(X):

     X_train, X_test = X[train_ix], X[test_ix]
     y_train, y_test = y[train_ix], y[test_ix]

 model = get_model(n_inputs, n_outputs)

 model.fit(X_train, y_train, verbose=0, epochs=100)
 # prediction on the test set
 yhat = model.predict(X_test)
 # round probabilities 
 yhat = yhat.round()
 # calculate accuracy
 acc = accuracy_score(y_test, yhat)
 
 print('>%.3f' % acc)
 results.append(acc)
 return results
 
# load dataset
X, y = get_dataset()
# evaluate 
results = evaluate_model(X, y)
# summarize 
print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))


# In[13]:


from numpy import asarray
from sklearn.datasets import make_multilabel_classification
from keras.models import Sequential
from keras.layers import Dense
 
def get_dataset():
 X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
 return X, y
 
# model
def get_model(n_inputs, n_outputs):
 model = Sequential()
 model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
 model.add(Dense(n_outputs, activation='sigmoid'))
 model.compile(loss='binary_crossentropy', optimizer='adam')
 return model
 
X, y = get_dataset()
n_inputs, n_outputs = X.shape[1], y.shape[1]

model = get_model(n_inputs, n_outputs)

model.fit(X, y, verbose=0, epochs=100)

#prediction for new data

row = [3, 3, 6, 7, 8, 2, 11, 11, 1, 3]
newX = asarray([row])
yhat = model.predict(newX)

print('Predicted: %s' % yhat[0])


# In[16]:


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


# In[17]:


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


# In[ ]:



import tensorflow_hub as hub

feature_extractor_url = "/Users/avni/Desktop/Pigeon"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))


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




