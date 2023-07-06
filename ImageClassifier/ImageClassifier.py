'''Used this tutorial:
https://towardsdatascience.com/multi-label-image-classification-in-tensorflow-2-0-7d4cf8a4bc72
'''
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

IMG_SIZE = 224 # Specify height and width of image to match the input format of the model
CHANNELS = 3 # Keep RGB color channels to match the input format of the model

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

BATCH_SIZE = 256 # Big enough to measure an F1-score
AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically to reduce GPU and CPU idle time
SHUFFLE_BUFFER_SIZE = 1024 # Shuffle the training data by a chunck of 1024 observations

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

'''  
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
'''

X_train = ["/home/ubuntu/peoplewelcome-group_10/ImageClassifier/pigeons_ds/Butterfly/001.jpg",
             "/home/ubuntu/peoplewelcome-group_10/ImageClassifier/pigeons_ds/Butterfly/002.jpg",
             "/home/ubuntu/peoplewelcome-group_10/ImageClassifier/pigeons_ds/Butterfly/003.jpg",
             "/home/ubuntu/peoplewelcome-group_10/ImageClassifier/pigeons_ds/Butterfly/004.jpg",
             "/home/ubuntu/peoplewelcome-group_10/ImageClassifier/pigeons_ds/Pigeon/001.jpg",
             "/home/ubuntu/peoplewelcome-group_10/ImageClassifier/pigeons_ds/Pigeon/002.jpg",
             "/home/ubuntu/peoplewelcome-group_10/ImageClassifier/pigeons_ds/Pigeon/003.jpg",
             "/home/ubuntu/peoplewelcome-group_10/ImageClassifier/pigeons_ds/Pigeon/004.jpg"]

y_train_bin = np.array([["Butterfly","Grass"],
                       ["Butterfly","Leaf"],
                       ["Butterfly","Leaf"],
                       ["Butterfly"],
                       ["Pigeon","Grass"],
                       ["Pigeon"],
                       ["Pigeon"],
                       ["Pigeon"]])

stock_images = [0 for i in range(10)] 
stock_images[0] = "pigeons_validate/Pigeon/5.jpg"
stock_images[1] = "pigeons_validate/Pigeon/Pigeon.jpeg"
stock_images[2] = "pigeons_validate/Pigeon/pigeons.jpg"
stock_images[3] = "pigeons_validate/Butterfly/1.jpg"
stock_images[4] = "pigeons_validate/Butterfly/2.png"
stock_images[5] = "pigeons_validate/Butterfly/3.jpg"
stock_images[6] = "pigeons_validate/Butterfly/is-this-a-pigeon.jpeg"

y_val_bin = np.array([["Pigeon"],
                       ["Pigeon"],
                       ["Pigeon"],
                       ["Butterfly"],
                       ["Butterfly"],
                       ["Butterfly"],
                       ["Butterfly"]])

train_ds = create_dataset(X_train, y_train_bin)
val_ds = create_dataset(stock_images, y_val_bin)


feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))

feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(1024, activation='relu', name='hidden_layer'),
    layers.Dense(N_LABELS, activation='sigmoid', name='output')
])

def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    
    return macro_cost

LR = 1e-5 # Keep it small when transfer learning
EPOCHS = 30

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
  loss=macro_soft_f1,
  metrics=[macro_f1])

history = model.fit(train_ds,
  epochs=EPOCHS,
  validation_data=create_dataset(X_val, y_val_bin))

#parse_function("/home/ubuntu/peoplewelcome-group_10/ImageClassifier/pigeons_ds/Butterfly/001".jpg, [Butterfly, Grass])