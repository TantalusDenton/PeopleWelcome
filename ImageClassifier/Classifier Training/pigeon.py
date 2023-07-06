import numpy as np
import PIL
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

stock_images = [0 for i in range(10)] 
stock_images[0] = "/Users/avni/Desktop/Pigeon/109\ 2.jpg"
stock_images[1] = "/Users/avni/Desktop/Pigeon/107\ 2.jpg"
stock_images[2] = "/Users/avni/Desktop/Pigeon/110\ 2.jpg"
stock_images[3] = "Users/avni/Desktop/Pigeon/108\ 2.jpg"
stock_images[4] = "/Users/avni/Desktop/Pigeon/112\ 2.jpg"
stock_images[5] = "Users/avni/Desktop/Pigeon/113\ 2.jpg"
stock_images[6] = "/Users/avni/Desktop/Pigeon/114\ 2.jpg"
stock_images[7] = "/Users/avni/Desktop/Pigeon/111\ 2.jpg"
stock_images[8] = "/Users/avni/Desktop/Pigeon/104.jpg "
stock_images[9] = "/Users/avni/Desktop/Pigeon/117.jpg"

checkpoint_path = "pigeons_model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data_dir = "pigeons_ds"
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180

labels = ['Butterfly', 'Pigeon']

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print("class_names at start ", class_names) 

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

global num_classes
num_classes = len(class_names)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

def createModel():
  print("class_names after add label ", class_names) 
  global model 
  model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  
  return model

def train():
  epochs = 15
  # Create a callback that saves the model's weights
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[cp_callback]
  )

def infer(img_num):
  pidgeon_path = stock_images[img_num]

  img = tf.keras.utils.load_img(
      pidgeon_path, target_size=(img_height, img_width)
  )

  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  print("predicting")
  print(predictions)

  label = class_names[np.argmax(score)]

  prediction = [label, score]

  if 100 * np.max(score) > 90.0 :
    print(
          "This image most likely shows a {} with a {:.2f} percent confidence."
          .format(class_names[np.argmax(score)], 100 * np.max(score)))
    return prediction
  else:
    print(
          "I'm not sure what this image shows ")
    return ['Not sure', score]

# Separate function for classifying a picture from URL. ToDo: this will be reading a DB of uploads instead of feching url
def inferUploadedImg(img_url):
  image_path = tf.keras.utils.get_file(os.path.basename(img_url), img_url)
  print(os.path.basename(img_url))

  img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  label = class_names[np.argmax(score)]

  prediction = [label, score]

  #If the confidence is low, then it is likely misclassified, discard.
  if 100 * np.max(score) > 90.0 :
    print(
          "This image most likely shows a {} with a {:.2f} percent confidence."
          .format(class_names[np.argmax(score)], 100 * np.max(score)))
    return prediction
  else:
    print(
          "I'm not sure what this image shows ")
    return ['Not sure', score]

def addLabel(newLabel):
  #train_ds.labels.append(newLabel) #ToDo: object has no attribute labels. Find a way to add labels to ds
  class_names.append(newLabel)
  global num_classes #ToDo: review Programming Languages var scopes lecture and verify this is correct way
  num_classes = num_classes + 1

def reTrain():
  model = createModel()
  train()

#addLabel("Cow")
model = createModel()

# If you have a trained model, loads the weights
#model.load_weights(checkpoint_path)

#If you don't have a trained model, train it and save
train()

#sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
#sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

test_url = "https://i.ytimg.com/vi/Kb3U7oKlIWI/maxresdefault.jpg"

inferUploadedImg(test_url)

predictions = [0 for i in range(10)] 
labels = [0 for i in range(10)] 
scores = [0 for i in range(10)] 

for i in range(10):
  predictions[i] = infer(i)
  labels[i] = predictions[i][0]
  # scores[i] = np.max(predictions[i][1])
