import keras
from keras.layers import Activation
from keras.layers import Dropout, Conv2D, BatchNormalization, Dense, Flatten, Reshape, MaxPooling2D


def get_imageRecognition_model(input_shape, activation_functions, padding, output_shape):
   # Neural network architecture for digit image recognition

    model = keras.models.Sequential()
    model.add((Conv2D(60,kernel_size=(5,5),input_shape=input_shape ,padding = padding ,activation = activation_functions[0])))
    model.add((Conv2D(60, kernel_size=(5,5),padding=padding,activation=activation_functions[0])))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add((Conv2D(30,kernel_size=(3,3),padding=padding, activation=activation_functions[0])))
    model.add((Conv2D(30, kernel_size= (3,3), padding=padding, activation=activation_functions[0])))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500,activation = activation_functions[0]))

    # model.add(Dense(500,activation = activation_functions[0]))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation = activation_functions[1]))

    return model


def get_Sudoku_model(input_shape, activation_functions, padding):

    model = keras.models.Sequential()

    model.add(Conv2D(64, kernel_size=(3,3), activation = activation_functions[0], padding=padding, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation = activation_functions[0], padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(1,1), activation = activation_functions[0], padding=padding))
    model.add(Flatten())
    model.add(Dense(81*9))
    model.add(Reshape((-1, 9)))
    model.add(Activation(activation_functions[1]))
    
    return model