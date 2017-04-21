from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K

from imageHandler import imageHandler

epoch_count = 10
learn_rate = 0.0001
data_folders = ["./data/straight-bridge",
                "./data/straight-corners",
                "./data/straight-recovery",
                "./data/straight-reverse",
                "./data/straight-smooth",
                "./data/straight-recovery-side"]

imgH = imageHandler(data_folders)
data_count = imgH.get_data_count()


def comma_model():
    model = Sequential()
    model.add(Convolution2D(16, 8, 8, input_shape=(160, 320, 3), subsample=(4, 4), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(.2))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    # model.add(Dense(32))
    # model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=0.0001), loss="mse", metrics=['accuracy'])
    return model

def nvidia_model():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, input_shape=(128, 128, 3), border_mode='same', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer=Adam(learn_rate), loss="mse")
    return model


model = nvidia_model()
train_samples = data_count[0]# - data_count[0]%128
val_samples = data_count[1] #- data_count[1]%128
hist = model.fit_generator(generator=imgH.get_training_batch(), samples_per_epoch=train_samples, nb_epoch=10,
                                     validation_data=imgH.get_validation_batch(), nb_val_samples=val_samples)

model.summary()

model.save_weights('model.h5')
json_string = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(json_string)

K.clear_session()
