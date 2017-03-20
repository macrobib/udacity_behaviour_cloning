import os
import cv2
import csv
import glob
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from sklearn.model_selection import train_test_split


class imageHandler:
    """Utility functions to handle images."""
    def __init__(self, data_folder=None):
        self.data_folder =  data_folder
        self.data_dict = {}
        self.data_count = None
        self.train = None
        self.validate = None
        self.steering = None
        self.validate_steering = None
        self.csv_files = []
        if data_folder != None:
            self.data_files = glob.glob(data_folder[0] + '/IMG/*.jpg')
            for folder in data_folder:
                self.csv_files.append(glob.glob(folder + '/driving_log.csv')[0])
            print(self.csv_files)

    def get_training_batch(self, batch_size=128):
        count = len(self.train)
        while 1:
            offset = 0
            for index in range(0, len(self.train), batch_size):
                offset = index + batch_size
                if offset <= count:
                    yield (self.train[index:offset], self.steering[index:offset])

    def get_validation_batch(self, batch_size=128):
        count = len(self.train)
        while 1:
            offset = 0
            for index in range(0, len(self.validate), batch_size):
                offset = index + batch_size
                if offset <= count:
                    yield (self.validate[index: offset], self.validate_steering[index: offset])

    def print_file_info(self):
        """Print info on the saved file."""
        val = [i for i in range(1000)]
        random.shuffle(val)
        val = val[:20]
        for i in val:
            image = mpimage.imread(self.data_files[i])
            # plt.imshow(image)
            # plt.show()

    def shuffle_and_split(self):
        """Shuffle and split data"""
        assert len(self.train) == len(self.steering)
        row, _, _, _ = self.train.shape
        index = np.arange(0, row)
        index = np.random.permutation(index)
        self.train = self.train[index, :, :, :]
        self.steering = self.steering[index]
        self.train, self.validate, self.steering, self.validate_steering = train_test_split(self.train, self.steering
                                                                                            , test_size=0.2,
                                                                                            random_state=42)

    def process_data(self):
        """Process image and csv and create a pickle file."""
        count = 0
        # Three different sets of data:- normal driving, curves only and counter-clockwise.
        for file_index in self.csv_files:
            print(file_index)
            file = open(file_index)
            reader = csv.reader(file)
            data = list(reader)
            for index, row in enumerate(data):
                self.data_dict[count] = {0: row[0], 1: row[3], 2: 'center'}
                self.data_dict[count + 1] = {0: row[1], 1: row[3], 2: 'left'}
                self.data_dict[count + 2] = {0: row[2], 1: row[3], 2: 'right'}
                count += 3
        print("Dictionary length", len(self.data_dict))

    def load_and_pickle_data(self, model_name='comma'):
        """"""
        list_train = []
        list_steering = []
        if os.path.isfile('../data/pickle/train.p'):
            with open('../data/pickle/train.p', 'rb') as f:
                data = pickle.load(f)
                self.train = data['images']
                self.steering = data['steering']
                # plt.imshow(self.train[0])
                # plt.show()
            with open('../data/pickle/validate.p', 'rb') as f:
                data = pickle.load(f)
                self.validate = data['images']
                self.validate_steering = data['steering']
                # plt.imshow(self.validate[0])
                # plt.show()
            self.data_count = (self.train.shape[0], self.validate.shape[0])
        else:
            print("creating pickle file")
            for key, val in self.data_dict.items():
                # img = mpimage.imread(val[0])
                img = cv2.imread(val[0])
                angle = float(val[1])
                # img = self.resize_image(img, model_name)

                if val[2] == 'left':
                    if angle + 0.01 <= 1.:
                        img = self.angle_correction(img, 0.01)
                        list_train.append(img)
                        list_steering.append(angle)
                elif val[2] == 'right':
                    if angle - 0.01 >= -1.:
                        img = self.angle_correction(img, -0.01)
                        list_train.append(img)
                        list_steering.append(angle)
                else:
                    list_train.append(img)
                    list_steering.append(angle)

                    img = self.augment_image(img)
                    list_train.append(img)
                    list_steering.append(angle)
                # self.check_image(img)

            # Convert to numpy array.
            self.train = np.array(list_train)
            self.steering = np.array(list_steering)
            self.shuffle_and_split()
            self.data_count = (self.train.shape[0], self.validate.shape[0])
            print(len(self.train))
            # Smoothen the steering angle values.
            # self.smoothen_steering_angles()
            with open('../data/pickle/train.p', 'wb') as f:
                data = {'images': self.train, 'steering': self.steering}
                pickle.dump(data, f)
            with open('../data/pickle/validate.p', 'wb') as f:
                data = {'images': self.validate, 'steering': self.validate_steering}
                pickle.dump(data, f)

    def resize_image(self, image, model, convert_to_grayscale=False):
        """Resize the image."""
        if convert_to_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if model == 'comma':
            img = cv2.resize(image, (160, 320), interpolation=cv2.INTER_AREA)
        elif model == 'nvidia':
            img = cv2.resize(image, (160, 320), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(image, (160, 320), interpolation=cv2.INTER_AREA) # VGG Model
        return img


    def angle_correction(self,img,  val):
        """Add angle correction values."""
        row, cols, _ = img.shape
        M = np.float32([[1, 0, val], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (cols, row))
        return dst


    def smoothen_steering_angles(self, n=9):
        """Average out the steering angles within -1 to 1 window."""
        print(self.steering)
        val = range(0, len(self.steering), n)
        for i in val:
            temp_sum = sum(self.steering[i:i+n])
            self.steering[i] = temp_sum/n
        print(self.steering)

    def augment_image(self, img):
        """Create perturbation to images."""
        row, cols, _ = img.shape
        val_x = random.uniform(-1.5, 1.5)
        val_y = random.uniform(-1.5, 1.5)
        M = np.float32([[1, 0, val_x], [0, 1, val_y]])
        dst = cv2.warpAffine(img, M, (cols, row))
        return dst

    def check_image(self, img):
        """Test function."""
        print("shape of image", img.shape[:2])
        # cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        cv2.imshow('main', img[20:, :20])
        cv2.waitKey(0)

    def get_data_count(self):
        return self.data_count





