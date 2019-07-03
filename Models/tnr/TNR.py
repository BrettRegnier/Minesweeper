# Convolutional Neural Network

#importing keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
import os

image_size = 32

# Initialize the CNN
classifier = Sequential()

classifier.add(Convolution2D(128, 3, 1, input_shape = (image_size, image_size, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

# Full connection
classifier.add(Dense(output_dim = 256, activation='relu'))
classifier.add(Dense(output_dim = 64, activation='relu'))
classifier.add(Dense(output_dim = 12, activation='softmax'))

# Compile CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
    
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
    'dataset/training',
    target_size = (image_size, image_size),
    batch_size = 32,
    class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(
    'dataset/test',
    target_size = (image_size, image_size),
    batch_size = 10,
    class_mode = 'categorical')

from IPython.display import display
from PIL import Image

if os.path.isfile('Models/TNR/TNR.h5'):
    classifier.load_weights('Models/TNR/TNR.h5')
else:
    classifier.fit_generator(
        training_set,
        steps_per_epoch=2000,
        epochs=3,
        validation_data=test_set,
        validation_steps=400)
        
    classifier.save_weights('Models/TNR/TNR.h5')

import numpy as np
from keras.preprocessing import image

from os import listdir
from os.path import isfile, join
path = 'dataset/gameset/'
files = [f for f in listdir(path) if isfile(join(path, f))]

for fl in files:
    # print(fl)
    test_image = image.load_img(path + fl, target_size = (image_size, image_size))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    
    # print(result)
    if result[0][0] == 1:
        print(1)
    elif result[0][1] == 1:
        print(2)
    elif result[0][2] == 1:
        print(3)
    elif result[0][3] == 1:
        print(4)
    elif result[0][4] == 1:
        print(5)
    elif result[0][5] == 1:
        print(6)
    elif result[0][6] == 1:
        print(7)
    elif result[0][7] == 1:
        print(8)
    elif result[0][8] == 1:
        print('flagged')
    elif result[0][9] == 1:
        print('mine')
    elif result[0][10] == 1:
        print('revealed')
    elif result[0][11] == 1:
        print('unrevealed')
    
