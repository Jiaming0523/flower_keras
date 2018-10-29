import numpy as np 
import pandas as pd 
from PIL import  Image

from PIL import Image
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras import backend

import os

print(os.listdir("/media/sf_Shared_F/flowers/"))

script_dir = os.path.dirname(".")
training_set_path = os.path.join(script_dir, '/media/sf_Shared_F/flowers')
test_set_path = os.path.join(script_dir, '/media/sf_Shared_F/flowers')

#building the CNN
classifier = Sequential() 
input_size = (256, 256)
classifier.add(Conv2D(32, (3,3), input_shape=(256,256,3), activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2,2))) 
classifier.add(Flatten()) 
classifier.add(Dense(units = 128, activation='relu')) 
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 32

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.33)

training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size=input_size,
                                                 batch_size=batch_size,
                                                 subset="training",
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=input_size,
                                            batch_size=batch_size,
                                            subset="validation",
                                            class_mode='categorical')

model_info = classifier.fit_generator(training_set,
                         steps_per_epoch=1000/batch_size,
                         epochs=10,
                         validation_data=test_set,
                         validation_steps=100/batch_size)

test_loss, test_accuracy = classifier.evaluate(test_datagen, test_set)
classifier.save('./classify.h5')
print(test_loss,test_accuracy)
