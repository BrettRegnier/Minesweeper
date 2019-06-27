# Convolutional Neural Network

#importing keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os

# Initialize the CNN
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(16, 3, 1, input_shape = (20, 20, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(output_dim = 128, activation='relu'))
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
    target_size = (20, 20),
    batch_size = 32,
    class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(
    'dataset/test',
    target_size = (20, 20),
    batch_size = 32,
    class_mode = 'categorical')

from IPython.display import display
from PIL import Image

if os.path.isfile('Multiclassifier.h5'):
    classifier.load_weights('Multiclassifier.h5')
else:
    classifier.fit_generator(
        training_set,
        steps_per_epoch=1600,
        epochs=5,
        validation_data=test_set,
        validation_steps=400)
        
    classifier.save_weights('Multiclassifier.h5')

import numpy as np
from keras.preprocessing import image

from os import listdir
from os.path import isfile, join
path = 'dataset/gameset/'
files = [f for f in listdir(path) if isfile(join(path, f))]

for fl in files:
    # print(fl)
    test_image = image.load_img(path + fl, target_size = (20, 20))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    print(result)
    
