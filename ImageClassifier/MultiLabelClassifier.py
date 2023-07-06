# MIT License
#
# Copyright (c) 2023 PeopleWelcome
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE

# prereqs:
# pip install -U scikit-learn
# pip install seaborn utils

import logging
import os
from random import randint, uniform
import time
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
from keras import layers

from utils import *
from utils import perf_grid as perfgrid
import ImageLogicQueries
import glob
import shutil
import time
from pathlib import Path
import SchedulerQueries
import re

warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)

print("TF version:", tf.__version__)

# get rid of the queue and make this file a class
# need a queue to put train requests

img_height = 224
img_width = 224

def cleanup():
  dataset1 = Path("./ai-data/ai-dataset.csv")
  if(dataset1.is_file()):
    os.remove(dataset1)
  dataset1 = Path("./ai-data/train-dataset.csv")
  if(dataset1.is_file()):
    os.remove(dataset1)
  dataset1 = Path("./ai-data/validation-dataset.csv")
  if(dataset1.is_file()):
    os.remove(dataset1)

  trainfiles = glob.glob('./ai-data/images/train/*')
  for f in trainfiles:
      os.remove(f)

  valfiles = glob.glob('./ai-data/images/validation/*')
  for f in valfiles:
      os.remove(f)

cleanup()

def addLabel(newLabel):
  with open('./ai-data/ai-dataset.csv','a') as fd:
    fd.write(newLabel+"|")

# read file, flatten, normalize, append to the dataset
def addImage(newImage, imgid):
  print("adding new image")
  img = tf.keras.utils.load_img(
        newImage, target_size=(img_height, img_width)
    )
  #img_array = tf.keras.utils.img_to_array(img)
  #print("img_array is ", img_array)
  img = img.save("./ai-data/images/train/" + imgid + ".jpg")

def augmentData(csvfilepath, imagespath):
  if csvfilepath is './ai-data/train-dataset.csv':
    shutil.copyfile('./ai-data/ai-dataset.csv', csvfilepath)
  else:
    with open(csvfilepath,'a') as fd:
      fd.write("image-id,tags\n") 

  originalDataset = pd.read_csv("./ai-data/ai-dataset.csv", encoding="ISO-8859-1")

  imageIds = originalDataset['image-id']
  imageIds = list(imageIds)

  tags = originalDataset['tags']
  tags = list(tags)

  i=0
  for imageId in imageIds:
    with open(csvfilepath,'a') as fd:
        for iteration in range(50):
          newImageId = augmentImage(imageId, iteration, imagespath)
          fd.write(newImageId+","+tags[i]+"\n")
        i = i+1

def augmentImage(originalImageId, iteration, imagespath):
   filepath = os.path.join("./ai-data/images/train/", str(originalImageId)+'.jpg')
   OriginalImage = Image.open(filepath)

   #rotate
   rotated_image = OriginalImage.rotate(randint(-10, 10))

   #zoom
   zoom = uniform(1,1.5)
   x = randint(100, 150)
   y = randint(100, 150)
   w, h = rotated_image.size
   zoom2 = zoom * 2
   rotated_image = rotated_image.crop((x - w / zoom2, y - h / zoom2, 
                  x + w / zoom2, y + h / zoom2))
   
   augged_image = rotated_image.resize((w, h), Image.LANCZOS)

   if(randint(0,1)):
    augged_image = augged_image.transpose(Image.FLIP_LEFT_RIGHT)

   auggedImageId = originalImageId + str(iteration)

   newImage = augged_image

   newImage.save(imagespath + auggedImageId + ".jpg")
   return auggedImageId

def getPWData(user, ai):
  print("inside getPWData..."+user,ai)
  #time.sleep(3) # Temporary hack. Make a training scheduler instead
  imagesAndTags = ImageLogicQueries.getImagesAndTags(user, ai)

  with open('./ai-data/ai-dataset.csv','a') as fd:
    fd.write("image-id,tags\n")

  for item in imagesAndTags:
    currImage = item['image_id']
    image = ImageLogicQueries.downloadImage(currImage)
    addImage(image, currImage)
    with open('./ai-data/ai-dataset.csv','a') as fd:
        fd.write(currImage+",")
    tags = item['tags'] # ToDo: implement a hashmap, or make sure no duplicates happen
    for tag in tags:
        #ToDo: if label contains ',' then drop this character (causes csv separation)
        print("adding label {0} to image {1}".format(tag,currImage))
        addLabel(re.sub(",","",tag) )
    with open('./ai-data/ai-dataset.csv','a') as fd:
        fd.write("\n")

  augmentData('./ai-data/train-dataset.csv','./ai-data/images/train/')
  augmentData('./ai-data/validation-dataset.csv','./ai-data/images/validation/')

def createDatasets():
  datasetTraining = pd.read_csv("./ai-data/train-dataset.csv", encoding="ISO-8859-1")
  datasetValidation = pd.read_csv("./ai-data/validation-dataset.csv", encoding="ISO-8859-1")

  # Removes rows with missing Id, or tags 
  datasetTraining.dropna(subset=['image-id','tags'], inplace=True)
  datasetValidation.dropna(subset=['image-id','tags'], inplace=True)

  # tags to array
  datasetTraining['tags'] = datasetTraining['tags'].apply(lambda s: [l for l in str(s).split('|') if l not in [""]])
  datasetValidation['tags'] = datasetTraining['tags'].apply(lambda s: [l for l in str(s).split('|') if l not in [""]])

  print(datasetTraining.head())

  global X_train
  X_train = datasetTraining['image-id']
  global X_val
  X_val = datasetValidation['image-id']
  X_train = [os.path.join('./ai-data/images/train/', str(f)+'.jpg') for f in X_train]
  X_val = [os.path.join('./ai-data/images/validation/', str(f)+'.jpg') for f in X_val]

  print("Number of images for training: ", len(X_train))
  print("Number of images for validation: ", len(X_val))

  y_train = datasetTraining['tags']
  y_val = datasetValidation['tags']
  y_train = list(y_train)
  y_val = list(y_val)

  print(y_train[:3])

  # Fit the multi-label binarizer on the training set
  print("Labels:")
  global mlb
  mlb = MultiLabelBinarizer()
  mlb.fit(y_train)

  # Loop over all labels and show them
  global N_LABELS
  N_LABELS = len(mlb.classes_)
  for (i, label) in enumerate(mlb.classes_):
      print("{}. {}".format(i, label))

  global y_train_bin
  y_train_bin = mlb.transform(y_train)
  global y_val_bin
  y_val_bin = mlb.transform(y_val)

  for i in range(3):
      print(X_train[i], y_train_bin[i])

IMG_SIZE = 224 # Specify height and width of image to match the input format of the model
CHANNELS = 3 

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

BATCH_SIZE = 256 # measure an F1-score
AUTOTUNE = tf.data.experimental.AUTOTUNE # preprocessing
SHUFFLE_BUFFER_SIZE = 1024

def create_dataset(filenames, labels, is_training=True):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    print("start create_dataset")
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Parse and preprocess observations in parallel

    for element in dataset.as_numpy_iterator(): 
      print(element) 

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
    
    print("end create_dataset")
    return dataset

def createModel():
  global train_ds
  train_ds = create_dataset(X_train, y_train_bin)
  global val_ds
  val_ds = create_dataset(X_val, y_val_bin)

  for f, l in train_ds.take(1):
      print("Shape of features array:", f.numpy().shape)
      print("Shape of labels array:", l.numpy().shape)

  feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
  feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                          input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))

  feature_extractor_layer.trainable = False

  global model
  model = tf.keras.Sequential([
      feature_extractor_layer,
      layers.Dense(1024, activation='relu', name='hidden_layer'),
      layers.Dense(N_LABELS, activation='sigmoid', name='output')
  ])

  model.summary()

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

def train():
  LR = 1e-5 # Keep it small when transfer learning
  EPOCHS = 30

  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=macro_soft_f1,
    metrics=[macro_f1])

  #start = time()
  history = model.fit(train_ds,
                      epochs=EPOCHS,
                      validation_data=create_dataset(X_val, y_val_bin))
  #print('\nTraining took {}'.format(print_time(time()-start)))

def reTrain(user, ai):
  #time.sleep(3) # Temporary hack. Make a training scheduler instead
  getPWData(user, ai)
  createDatasets()
  createModel()
  train()

'''# Get all label names
label_names = mlb.classes_
# Performance table with the first model (macro soft-f1 loss)
grid = perfgrid(val_ds, y_val_bin, label_names, model)'''

def infer():
  img_path = "./ai-data/images/cat1.jpg"

  # Read and prepare image
  img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE,CHANNELS)) # if just 'image.load_img' then error: AttributeError: module 'keras.preprocessing.image' has no attribute 'load_img'
  img = tf.keras.utils.img_to_array(img)
  img = img/255
  img = np.expand_dims(img, axis=0)

  # Generate prediction
  prediction = (model.predict(img) > 0.5).astype('int')
  prediction = pd.Series(prediction[0])
  prediction.index = mlb.classes_
  prediction = prediction[prediction==1].index.values

  print("predicted tags: \n")
  print(list(prediction))
  return list(prediction)

while True:
  time.sleep(3)
  dataFromQueue = SchedulerQueries.getFirstElementInQueue()
  if dataFromQueue[1] != 'empty':
    print("not empty \n")
    print("the dataFromQueue is:")
    print(dataFromQueue)
    retrainThisUser = dataFromQueue[0]
    retrainThisAi = dataFromQueue[1]
    reTrain(retrainThisUser, retrainThisAi)
    cleanup()
    class_names = infer()