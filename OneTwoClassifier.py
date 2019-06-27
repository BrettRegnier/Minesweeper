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
classifier.add(Convolution2D(32, 3, 3, input_shape = (20, 20, 3), activation = 'relu'))

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
    'dataset/training',
    target_size = (20, 20),
    batch_size = 32,
    class_mode = 'binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test',
    target_size = (20, 20),
    batch_size = 32,
    class_mode = 'binary')

from IPython.display import display
from PIL import Image

if os.path.isfile('OneTwoWeights.h5'):
    classifier.load_weights('OneTwoWeights.h5')
else:
    classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=10,
        validation_data=test_set,
        validation_steps=800)
        
    classifier.save_weights('OneTwoWeights.h5')
    
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/2algerian.jpg', target_size = (20, 20))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] >= 0.5:
    prediction = "2"
else:
    prediction = "1"
    
print(prediction)