

import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os




filenames = os.listdir('/Users/avni/Desktop/cats_dogs_cars/data')

categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'car': 
        categories.append(0)  #The categories are set as labels
    elif category == 'cat': 
        categories.append(1)
    elif category == 'dog':
        categories.append(2)
           




df = pd.DataFrame({
    'filename': filenames,  
    'category': categories, })


# In[9]:


df


# In[13]:


df['category'].value_counts().plot.bar()
print(df['category'].value_counts())

print("\n Total images: ", df.shape[0])


# In[15]:


sample = random.choice(filenames)
image = load_img('/Users/avni/Desktop/cats_dogs_cars/data/'+sample)
plt.imshow(image)


# In[16]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model2 = Sequential()

model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150,150, 3)))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(96, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(3, activation='softmax')) # 3 because we have car, dog and cat classes

model2.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model2.summary()


# In[17]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001) 
callbacks = [earlystop, learning_rate_reduction]


df["category"] = df["category"].replace({ 0: 'car', 1: 'cat', 2: 'dog' })




df["category"].head()




train_df, validate_df = train_test_split(df, test_size=0.20, random_state=0) 

#total 'TRAIN' files and in each category
train_df = train_df.reset_index(drop=True)

train_df['category'].value_counts().plot.bar()
print(train_df['category'].value_counts())
print("\n Total train images: ", train_df.shape[0])


# In[21]:


validate_df = validate_df.reset_index(drop=True)
validate_df['category'].value_counts().plot.bar()

print(validate_df['category'].value_counts())
print("\n Total validate images: ", validate_df.shape[0])


# In[22]:


total_train = train_df.shape[0]
print("Total train images:", total_train)

total_validate = validate_df.shape[0] 
print("Total validate images:", total_validate)


# In[23]:


train_datagen = ImageDataGenerator(
                                    rotation_range=15,
                                    rescale=1./255,
                                    shear_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/Users/avni/Desktop/cats_dogs_cars/data", 
    x_col='filename',
    y_col='category',
    target_size=(150,150),
    class_mode='categorical',
    batch_size=32
)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "/Users/avni/Desktop/cats_dogs_cars/data", 
    x_col='filename',
    y_col='category',
    target_size=(150,150),
    class_mode='categorical',
    batch_size=32
)


# In[24]:


plt.figure(figsize=(12, 12))
for i in range(0, 12):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in train_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# In[25]:


epochs=50
batch_size=50

history = model2.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate/batch_size,
    steps_per_epoch=total_train/batch_size,
    callbacks=callbacks
)


# In[27]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,6))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))
ax1.legend()

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))
ax2.legend()

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# In[28]:


from keras import models    
model = models.load_model('catdogcar2.hdf5', compile = False)


# In[29]:


test_filenames = os.listdir("/Users/avni/Desktop/cats_dogs_cars/test")
test_df = pd.DataFrame({
    'filename': test_filenames
})


# In[30]:


total_test = test_df.shape[0]  
print("Total amount of test images are:", total_test)


# In[31]:


test_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 15

test_generator = test_datagen.flow_from_dataframe(
    test_df, 
    "/Users/avni/Desktop/cats_dogs_cars/test",
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=(150,150),
    batch_size=batch_size,
    shuffle=False 
)


# In[32]:


predict = model.predict_generator(test_generator, steps=np.ceil(total_test/batch_size))

#predicteed images       

print("Number of predicted images:", len(predict), "\n")

# Get predictions in array
# labels 0, 1, 2 (category)

print("    0:car          1:cat       2:dog \n\n", predict, "\n")   




test_df['category'] = np.argmax(predict, axis=-1) 

# predicted values for each category of images

print(test_df['category'][0:3]) 




print(train_generator.class_indices) 
print(validation_generator.class_indices, "\n")

# Map  
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
print(label_map, "\n")

# output of mapped labels
print(test_df['category'][0:4], "\n")

# replace letters with words
print(test_df['category'][0:3])


# In[35]:


test_df['category'].value_counts()
print(test_df['category'].value_counts())

# data in a bar graph
test_df['category'].value_counts().plot.bar()

# total predicted images 
print("\nTotal predicted images:", test_df.shape[0])


# In[36]:


sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("/Users/avni/Desktop/cats_dogs_cars/test/"+filename, target_size=(150,150)) #rescale(1./255)
    plt.subplot(9, 3, index+1)      
    plt.imshow(img)
    plt.xlabel(filename + "\n Predicted is " + "{}".format(category))
plt.tight_layout()
plt.show()


