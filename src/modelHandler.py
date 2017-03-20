import os
import json
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, ELU
from keras.layers import Flatten, BatchNormalization, Cropping2D
from keras.layers import Lambda, Input
from keras.backend import tf as ktf


class model:
    """Create/Save/Load training models"""

    def __init__(self, model_path=None, weight_name = None, json_name = None):
        self.model_weight_path = model_path + weight_name
        self.model_json_path = model_path + json_name
        self.model = None 
        self.model_name = None
        self.weights = None
        self.json = None
        self.default_path = "../model"

    def load_model(self):
        """Load a given model. Default model is VGG """
        print(self.model_json_path)
        with open(self.model_json_path, 'r') as f:
            string = f.read()
            val = json.loads(string)
            print(type(val))
            self.model = model_from_json(val)
        self.model.load_weights(self.model_weight_path)

    def save_model(self):
        """Save the model."""
        if os.path.exists(self.model_weight_path):
            os.remove(self.model_weight_path)

        self.model.save_weights(self.model_weight_path)
        json_string = self.model.to_json()

        if os.path.exists(self.model_json_path):
            os.remove(self.model_json_path)

        with open(self.model_json_path, 'w') as json_file:
            json_file.write(json_string)
        print("model saved to disk...")

    def create_comma_model(self, shape):
        """Create comma.ai model."""
        model = Sequential()
        #crop image.
        # model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
        # inp = Input(None, None, 3)
        # model.add(Lambda(lambda x: ktf.image.resize_images(x, (shape[0], shape[1]))))

        # Input image of the form row, column and channel.
        print("shape is: ", shape)
        # model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(shape[0], shape[1], 3),
        #                  output_shape=(shape[0], shape[1], 3)))
        model.add(BatchNormalization(input_shape=(shape[0], shape[1], 3), axis=1))
        model.add(ELU())
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same'))
        model.add(ELU())
        model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same'))
        model.add(ELU())
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same'))
        
        model.add(Flatten())
        model.add(Dropout(.2))
        model.add(ELU())
        model.add(Dense(512))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(1))
        opt = Adam()
        model.compile(loss='mean_squared_error', optimizer=opt)
        self.model = model

        return model

    def vgg_model(self):
        """ Pretrained VGG based model. """
        pass

    def create_nvidia_model(self, shape):
        """Create model based on end to end learning."""
        model = Sequential()
        model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
        model.add(Lambda(lambda x: ktf.image.resize_images(x, (shape[0], shape[1]))))
        model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(3, shape[0], shape[1]),
                         output_shape=(3, shape[0], shape[1])))
        model.add(BatchNormalization(epsilon=0.001, mode=2, axis=1, input_shape=(2, shape[0], shape[1])))
        model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
        model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
        model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
        model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
        model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))

        model.add(Flatten())
        model.add(Dense(1164, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.summary()

        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model


    def create_model(self, model_name):
        """Create model. If already exists load the model."""
        full_path = self.default_path + model_name
        if os.path.isfile(full_path):
            self.model_name = model_name
            self.load_model(model_name)
        else:
            if model_name == 'nvidia':
                shape = [128, 128]
                self.model_name = model_name
                self.create_nvidia_model(shape)
            elif model_name == 'comma':
                shape = [160, 320]
                self.model_name = model_name
                self.create_comma_model(shape)
            else:
                pass # Raise exception.
        return self.model

    def print_model(self):
        """Print the model structure."""
        self.model.summary()








