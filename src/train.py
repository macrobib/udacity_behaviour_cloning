import tensorflow as tf
from imageHandler import imageHandler
from modelHandler import model
import matplotlib.pyplot as plt

# Storage parameters.
MODEL_SELECT = 'comma'
MODEL_PATH = '../data/models/'
DATA_PATH = ['../data/track-3']#, '../data/track-4', '../data/track-5']
WEIGHT_NAME = 'model.h5'
MODEL_NAME = 'model.json'

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

# Load the model if necessary.
def main():

    # Process data.
    obj = imageHandler(DATA_PATH)
    obj.print_file_info()
    obj.process_data()
    obj.load_and_pickle_data()
    count = obj.get_data_count()  # count(<Number of training data points>, <No of validation data points.>)

    # Train Model
    model_obj = None
    model_to_fit = None
    if MODEL_SELECT == 'vgg':
        model_obj = model(MODEL_PATH, WEIGHT_NAME, MODEL_NAME)
        model_obj.load_model()
        model_obj.print_model()
    elif MODEL_SELECT == 'comma':
        model_obj = model(MODEL_PATH, 'comma.h5', 'comma.json')
        model_to_fit = model_obj.create_model('comma')
    else:
        model_obj = model(MODEL_PATH, 'nvidia.h5', 'nvidia.json')
        model_to_fit = model_obj.create_model('comma')

    # Fit Model.
    print(count)
    hist = model_to_fit.fit_generator(generator=obj.get_training_batch(), samples_per_epoch=count[0], nb_epoch=10,
                                     validation_data=obj.get_validation_batch(), nb_val_samples=count[1])

    print(hist.history.keys())
    model_obj.print_model()

    # Visualize the training.
    plot_history(hist)
    # Save the model
    model_obj.save_model()

if __name__ == '__main__':
    main()

