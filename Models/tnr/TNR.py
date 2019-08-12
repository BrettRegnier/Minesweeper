# Convolutional Neural Network
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
import os

class Network:
    def __init__(self):
        self._image_size = 32

        # Initialize the CNN
        self._classifier = Sequential()

        self._classifier.add(Conv2D(128, (3, 1), input_shape=(self._image_size, self._image_size, 3), activation="relu"))
        self._classifier.add(MaxPooling2D(pool_size=(3, 3)))
        self._classifier.add(Conv2D(64, (3, 1), input_shape=(self._image_size, self._image_size, 3), activation="relu"))
        self._classifier.add(MaxPooling2D(pool_size=(2, 2)))

        self._classifier.add(Flatten())

        # Full connection
        self._classifier.add(Dense(units = 256, activation='sigmoid'))
        self._classifier.add(Dense(units = 64, activation='sigmoid'))
        self._classifier.add(Dense(units = 12, activation='softmax'))

        # Compile CNN
        self._classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    def Train(self):
        # Fitting the CNN to` the images
        from keras.preprocessing.image import ImageDataGenerator

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
            
        test_datagen = ImageDataGenerator(rescale=1./255)
        training_set = train_datagen.flow_from_directory(
            'dataset/training',
            target_size = (self._image_size, self._image_size),
            batch_size = 32,
            class_mode = 'categorical')

        test_set = test_datagen.flow_from_directory(
            'dataset/test',
            target_size = (self._image_size, self._image_size),
            batch_size = 10,
            class_mode = 'categorical')
            
        self._classifier.fit_generator(
            training_set,
            steps_per_epoch=2000,
            epochs=10,
            validation_data=test_set,
            validation_steps=400)
            
        self._classifier.save_weights('Models/TNR/TNR.h5')
        
    def Predict(self):
        if os.path.isfile('Models/TNR/TNR.h5'):
            self._classifier.load_weights('Models/TNR/TNR.h5')
        else:
            self.Train()
        
        import numpy as np
        from keras.preprocessing import image

        from os import listdir
        from os.path import isfile, join
        path = 'dataset/gameset/'
        files = [f for f in listdir(path) if isfile(join(path, f))]

        for fl in files:
            # print(fl)
            test_image = image.load_img(path + fl, target_size = (self._image_size, self._image_size))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = self._classifier.predict(test_image)
            
            # TODO make this use the expected class names that from the folders.
            # print(result)
            if result[0][0] > 0.5:
                print("1, probability: " + str(result[0][0]))
            elif result[0][1] > 0.5:
                print("2, probability: " + str(result[0][1]))
            elif result[0][2] > 0.5:
                print("3, probability: " + str(result[0][2]))
            elif result[0][3] > 0.5:
                print("4, probability: " + str(result[0][3]))
            elif result[0][4] > 0.5:
                print("5, probability: " + str(result[0][4]))
            elif result[0][5] > 0.5:
                print("6, probability: " + str(result[0][5]))
            elif result[0][6] > 0.5:
                print("7, probability: " + str(result[0][6]))
            elif result[0][7] > 0.5:
                print("8, probability: " + str(result[0][7]))
            elif result[0][8] > 0.5:
                print("flagged, probability: " + str(result[0][8]))
            elif result[0][9] > 0.5:
                print("mine, probability: " + str(result[0][9]))                
            elif result[0][10] > 0.5:
                print("revealed, probability: " + str(result[0][10]))                
            elif result[0][11] > 0.5:
                print("unrevealed, probability: " + str(result[0][11]))                
                
if __name__ == "__main__":
    model = Network()
    model.Predict()