# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class less_ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(32, (3, 3), 
                  padding="same",
                  input_shape=inputShape))
        model.add(Activation("relu"))
        
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(classes, activation='softmax'))

        # return the constructed networeeeeehitecture
        return model
