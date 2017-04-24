import cv2
import csv
import math
import glob
import tqdm
import numpy as np
import random
import scipy.misc as mc
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from sklearn.model_selection import train_test_split
model_name = 'nvidia'

class imageHandler:
    """Utility functions to handle images."""
    def __init__(self, data_folders=None, grayscale = False):
        self.data_folders =  data_folders
        self.data_dict = {}
        self.data_count = None
        self.grayscale = False
        self.train = {}
        self.validate = {}
        self.steering = None
        self.validate_steering = None
        self.csv_files = []
        self.str_queue = deque(maxlen=4)
        self.pos_str_queue = deque(maxlen=5)
        if data_folders:
            count = 0
            data = []
            for folder in data_folders:
                print(folder)
                csv_file = glob.glob(folder + '/driving_log.csv')[0]
                with open(csv_file, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        row = list(row)
                        data.append({0:'center', 1:row[0], 2:row[3]})
                        data.append({0: 'left', 1: row[1], 2: row[3]})
                        data.append({0: 'right', 1: row[2], 2: row[3]})
                        count += 3
            data_len = len(data)
            random.shuffle(data)
            train_len = math.floor(data_len * 0.8)
            self.train = data[: train_len]
            self.validate = data[train_len:]
            self.data_count = (train_len, data_len - train_len)

    def crop_and_resize(self, img, model='comma'):
        img = img[50:, :]
        if model == 'nvidia':
            img = mc.imresize(img, (66, 200))
        else:
            img = mc.imresize(img, (160, 320))
        return img

    def flip_image(self, img):
        """Flip the given image by 180 degree"""
        img = np.fliplr(img)
        return img

    def convert_grayscale(self, img):
        return np.mean(img, axis=2, keepdims=True)

    def process_batch(self, index, offset, datatype):
        """Load and process set of data."""
        x_data = []
        y_data = []
        global model_name
        data_len = offset - index
        rand_index = None
        img_data = None
        choice = random.randint(0, 2)
        if datatype == 'train':
            img_data = self.train[index:offset]
            rand_index = [i for i in range(len(img_data))]
            random.shuffle(rand_index)
            rand_index = rand_index[0:(len(img_data) // 2)]  # 50% of the current batch size (128)
        elif datatype == 'valid':
            img_data = self.validate[index:offset]
        else:
            raise ValueError
        for row in img_data:
            img = None
            if self.grayscale:
                img = cv2.imread(row[1], 0)
            else:
                img = cv2.imread(row[1])
            steer_angle = float(row[2])

            if datatype == 'train':
                if choice == 0:
                    img, steer_angle = self.random_shear(img, steer_angle)
                elif choice == 1:
                    img = self.flip_image(img)
                    steer_angle = -steer_angle

            if row[0] == 'left':
                steer_angle += 0.1
                img = self.crop_and_resize(img, model_name)
            elif row[0] == 'right':
                steer_angle -= 0.1
                img = self.crop_and_resize(img, model_name)
            else:
                img = self.crop_and_resize(img, model_name)
            if img is None:
                raise ValueError
            img = img/127.5 - 1.0
            x_data.append(img)
            y_data.append(steer_angle)
        if datatype == 'train':
            for index in rand_index:
                if choice == 2:
                        x_data[index], y_data[index] = self.jitter_image_rotation(x_data[index], y_data[index])
        return np.asarray(x_data), np.asarray(y_data)

    def get_training_batch(self, batch_size=128):
        count = len(self.train)
        while 1:
            for index in range(0, count, batch_size):
                offset = index + batch_size
                x_train, y_train = self.process_batch(index, offset, 'train')
                if offset <= count:
                    yield (x_train, y_train)

    def get_validation_batch(self, batch_size=128):
        count = len(self.validate)
        while 1:
            for index in range(0, count, batch_size):
                offset = index + batch_size
                x_valid, y_valid = self.process_batch(index, offset, 'valid')
                if offset <= count:
                    yield (x_valid, y_valid)

    def shuffle_and_split(self):
        """Shuffle and split data"""
        assert len(self.train) == len(self.steering)
        row, _, _, _ = self.train.shape
        index = np.arange(0, row)
        index = np.random.permutation(index)
        self.train = self.train[index, :, :]
        self.steering = self.steering[index]
        self.train, self.validate, self.steering, self.validate_steering = train_test_split(self.train, self.steering
                                                                                            , test_size=0.2,
                                                                                            random_state=42)

    def angle_correction(self,img,  val):
        """Add angle correction values."""
        row, cols, _ = img.shape
        M = np.float32([[1, 0, val], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (cols, row))
        dst = dst.reshape((row, cols, 1))
        return dst

    def smoothen_steering_angles(self, n=9):
        """Average out the steering angles within -1 to 1 window."""
        print(self.steering)
        val = range(0, len(self.steering), n)
        for i in val:
            temp_sum = sum(self.steering[i:i+n])
            self.steering[i] = temp_sum/n
        print(self.steering)

    @staticmethod
    def augment_image(img):
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

    def movingaverage(self, interval, window_size = 5):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, 'same')

    def get_data_count(self):
        return self.data_count

    def jitter_image_rotation(self, image, steering):
        """Translation based jitter to image."""
        #https://github.com/vxy10/ImageAugmentation
        rows, cols, _ = image.shape
        range = 100
        numPixels = 10
        valPixels = 0.4
        transX = range * np.random.uniform() - range / 2
        steering +=  transX / range * 2 * valPixels
        transY = numPixels * np.random.uniform() - numPixels / 2
        transMat = np.float32([[1, 0, transX], [0, 1, transY]])
        image = cv2.warpAffine(image, transMat, (cols, rows))
        return image, steering

    def random_shear(self, image, steering_angle, shear_range=200):
        """
         https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
        """
        rows, cols, ch = image.shape
        dx = np.random.randint(-shear_range, shear_range + 1)
        random_point = [cols / 2 + dx, rows / 2]
        pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
        pts2 = np.float32([[0, rows], [cols, rows], random_point])
        dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
        steering_angle += dsteering
        return image, steering_angle






