import os
import cv2
import csv
import glob
import pickle
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimage


class imageHandler:
    """Utility functions to handle images."""
    def __init__(self, data_folder=None):
        self.data_folder =  data_folder
        self.data_dict = {}
        self.train = []
        self.steering = []
        self.validate = []
        if data_folder != None:
            self.data_files = glob.glob(data_folder + '/IMG/*.jpg')
            self.csv_file = glob.glob(data_folder + '/' + '*.csv')

    def print_file_info(self):
        """Print info on the saved file."""
        val = [i for i in range(1000)]
        random.shuffle(val)
        val = val[:20]
        for i in val:
            image = mpimage.imread(self.data_files[i])
            # plt.imshow(image)
            # plt.show()

    def process_data(self):
        """Process image and csv and create a pickle file."""
        file =  open(self.csv_file[0])
        reader = csv.reader(file)
        data = list(reader)
        # df = pd.read_csv(self.csv_file[0])
        columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        for index, row in enumerate(data):
            self.data_dict[index] = [row[0], row[3]]
            self.data_dict[index + 1] = [row[1], row[3]]
            self.data_dict[index + 2] = [row[2], row[3]]
        print(len(self.data_dict))

    def load_and_pickle_data(self, model_name='comma'):
        """"""
        if os.path.isfile('../data/pickle/train.p'):
            with open('../data/pickle/train.p', 'rb') as f:
                data = pickle.load(f)
                self.train = data['images']
                self.steering = data['steering']
                plt.imshow(self.train[0])
                plt.show()
        else:
            print("creating pickle file")
            for key, val in self.data_dict.items():
                # img = mpimage.imread(val[0])
                img = cv2.imread(val[0])
                img = self.resize_image(img, model_name)
                print("shape of image", img.shape[:2])
                cv2.imshow('img', img)
                cv2.waitKey(0)
                # self.train.append(img)
                # self.steering.append(val[1])
            # with open('../data/pickle/train.p', 'wb') as f:
            #     data = {'images': self.train, 'steering': self.steering}
            #     pickle.dump(data, f)
            print(type(img))

    def resize_image(self, image, model, convert_to_grayscale=False):
        """Resize the image."""
        if model == 'comma':
            img = cv2.resize(image, (160, 320), interpolation=cv2.INTER_AREA)
        elif model == 'nvidia':
            img = cv2.resize(image, (160, 320), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(image, (160, 320), interpolation=cv2.INTER_AREA) # VGG Model
        return img



    def augment_data(selfs):
        """Create perturbation to images."""






