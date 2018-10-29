import numpy as np 
import pandas as pd 
from PIL import  Image

from PIL import Image
# Importing the Keras libraries and packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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
for layers in vgg16_model.layers:
    model.add(layers)
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dense(4096,activation='relu'))
for layer in model.layers:
    layer.trainable=False
model.add(Dense(5,activation='softmax'))

optimizer=RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)

model.compile(optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
learning_rate_reduction=ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,minlr=0.00001)

batch_size = 32
train_datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.1,
        height_shift_range=0.1,  
        horizontal_flip=False,
        vertical_flip=False) 

test_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.33)

training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size=input_size,
                                                 batch_size=batch_size,
                                                 subset="training",
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=input_size,
                                            batch_size=batch_size,
                                            subset="validation",
                                            class_mode='binary')

model_info = classifier.fit_generator(training_set,
                         steps_per_epoch=1000/batch_size,
                         epochs=10,
                         validation_data=test_set,
                         validation_steps=100/batch_size)

test_loss, test_accuracy = classifier.evaluate(test_set, test_datagen)
classifier.save('./classify.h5')
print(test_loss,test_accuracy)
