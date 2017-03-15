import os
import json
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D
from keras.layers import Flatten, BatchNormalization

class model:
    """Create/Save/Load training models"""

    def __init__(self, weight_path=None, weight_name, json_name):
        self.model_weight_path = model_path + weight_name
        self.model_json_path = model_path + json_name
        self.model = None 
        self.model_name = None
        self.weights = None
        self.json = None
        self.default_path = "../models"

    def load_model(self, name):
        """Load a given model. Default model is VGG """
        self.model = model_from_json(self.model_json_path)
        self.model.load_weights(self.model_weight_path)


    def create_comma_model(self):
        """Create comma.ai model."""
        model = Sequential()
        model.add(Lambda(lambda: x:x/127.5 - 1., input_shape=(3, 160, 320)
                , output_shape=(3, 160, 320)))
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

        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model


    def create_nvidia_model(self):
        """Create model based on end to end learning."""
        model = Sequential()
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
        self.model = model

        #save model
        with open('../models/nvidia_model.json', 'w') as f:
            f.write(json.dumps(model.to_json()), indent=2)

        return model


    def create_model(self, model_name):
        """Create model. If already exists load the model."""
        full_path = self.default_path + model_name
        if os.path.isfile(full_path):
            self.model_name = model_name
            self.load_model(model_name)
        else:
            if model_name == 'nvidia':
                self.model_name = model_name
                self.create_nvidia_model()
            elif model.name == 'comma':
                self.model_name = model_name
                self.create_comma_model()
            else:
                pass # Raise exception.
        return self.model

    def print_model(self, model):
        """Print the model structure."""
        pass








