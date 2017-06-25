from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, MaxPooling2D, ELU, BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping


from imageHandler import imageHandler

epoch_count = 10
learn_rate = 0.0001
data_folders = [
                # "./data/straight-bridge",
                "./data/bridge-turn",
                # "./data/track-3",
                # "./data/straight-corners",
                "./data/straight-recovery",
                "./data/center",
                "./data/recover-unpatched",
                "./data/wiggle",#,
                # "./data/straight-reverse"]#,
                "./data/straight-smooth"]#,
                 # "./data/straight-recovery-side"]

imgH = imageHandler(data_folders)
data_count = imgH.get_data_count()

# plot history
def plot_history(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def comma_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(160, 320, 3)))
    # model.add(Convolution2D(16, 8, 8,input_shape=(160, 320, 3), subsample=(4, 4), border_mode="same"))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Dropout(.5))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Dropout(.5))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(ELU())
    model.add(Dropout(.2))
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=0.0001), loss="mse")
    return model


def nvidia_model():
    model = Sequential()

    # model.add(BatchNormalization(input_shape=(66, 200, 3)))
    # model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Convolution2D(24, 5, 5, input_shape=(66, 200, 3), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    # model.add(Dropout(0.2))

    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    # model.add(Dropout(0.2))

    model.add(Convolution2D(48, 3, 3, border_mode='same', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    # model.add(Dropout(0.4))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    # model.add(Dropout(0.4))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(1164))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learn_rate), loss="mse")
    model.summary()
    return model


def nvidia_model_2():
    model = Sequential()
    # model.add(BatchNormalization(input_shape=(66, 200, 3)))
    # model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Convolution2D(24, 5, 5, input_shape=(66, 200, 3), border_mode='same',
                            subsample=(2, 2), kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2),
                            kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.4))

    model.add(Convolution2D(48, 3, 3, border_mode='same', subsample=(2, 2),
                            kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.4))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1),
                            kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.4))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1),
                            kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    #
    model.add(Flatten())
    model.add(Dense(1164, kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learn_rate), loss="mse")
    model.summary()
    return model

# model = comma_model()
model = nvidia_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=2, min_delta=0.0002, mode='min')
train_samples = data_count[0] - data_count[0]%128
val_samples = data_count[1] - data_count[1]%128
hist = model.fit_generator(generator=imgH.get_training_batch(), samples_per_epoch=train_samples, nb_epoch=20,
                                     validation_data=imgH.get_validation_batch(), nb_val_samples=val_samples, callbacks=[early_stopping])

plot(model, to_file='model.png', show_shapes=True)
model.summary()

model.save_weights('model.h5')
json_string = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(json_string)

plot_history(hist)
K.clear_session()
