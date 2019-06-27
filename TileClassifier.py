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
classifier.add(Convolution2D(64, 3, 1, input_shape = (64, 64, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 1, activation='sigmoid'))

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
    'dataset/tile/training',
    target_size = (64, 64),
    batch_size = 10,
    class_mode = 'binary')

test_set = test_datagen.flow_from_directory(
    'dataset/tile/test',
    target_size = (64, 64),
    batch_size = 10,
    class_mode = 'binary')

from IPython.display import display
from PIL import Image

if os.path.isfile('BinaryClass.h5'):
    classifier.load_weights('BinaryClass.h5')
else:
    classifier.fit_generator(
        training_set,
        steps_per_epoch=100,
        epochs=1,
        validation_data=test_set,
        validation_steps=80)
        
    classifier.save_weights('BinaryClass.h5')
    
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/tile/unrevealed.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
if result[0][0] >= 0.5:
    prediction = "2"
else:
    prediction = "1"
    
print(prediction)