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
from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras import backend
from keras.optimizers import Adam

import os

print(os.listdir("/media/sf_Shared_F/flowers/"))

script_dir = os.path.dirname(".")
training_set_path = os.path.join(script_dir, '/media/sf_Shared_F/flowers')
test_set_path = os.path.join(script_dir, '/media/sf_Shared_F/flowers')

classifier = Sequential()
input_size = (256, 256)
classifier.add(Conv2D(32, (3, 3), input_shape=(256,256,3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=5, activation='softmax'))

opt = Adam(lr=1e-3, decay=1e-6)
classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

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
                         epochs=90,
                         validation_data=test_set,
                         validation_steps=100/batch_size,
                         workers=12)

from IPython.display import Image, display 
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

#def read_and_prep_images(img_paths, img_height, img_width): 
#	imgs = [load_img(img_path, target_size=(img_height, img_width)) 	for img_path in img_paths] 
#	    return np.array([img_to_array(img) for img in imgs])

#test_data = read_and_prep_images(image_names[0:10]) preds = model_1.predict(test_data)

#for i, img_path in enumerate(image_names): display(Image(img_path)) print(preds[i])
