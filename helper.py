import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image

from tqdm import tqdm
from os.path import isfile, isdir
from urllib.request import urlretrieve
import tarfile
import zipfile
import os
import random

from sklearn.utils import shuffle

width = 800
top = 220
bottom = 540
height = bottom - top
classes = 3

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def download_data():
    image_data = "data.tar.gz"

    if not isdir("data"): 
        if not isfile(image_data):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc='image data') as pbar:
                urlretrieve(
                    'https://storage.googleapis.com/fkhan-udacity/data.tar.gz',
                    image_data,
                    pbar.hook)

        targz = tarfile.open(image_data, 'r')
        targz.extractall()
        targz.close()

def download_data_backup():
    image_data = "lyft_training_data.tar.gz"
    image_data2 = "calra-capture-20180528.zip"

    if not isdir("Train"): 
        if not isfile(image_data):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc='image data') as pbar:
                urlretrieve(
                    'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz',
                    image_data,
                    pbar.hook)
        if not isfile(image_data2):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc='image data') as pbar:
                urlretrieve(
                    'https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180528.zip',
                    image_data2,
                    pbar.hook)

        targz = tarfile.open(image_data, 'r')
        targz.extractall()
        targz.close()

        zipdata = zipfile.ZipFile(image_data2, 'r')
        zipdata.extractall("Train")
        zipdata.close()

def getDataSetName():
    return os.listdir("data/Train/CameraRGB/")

def DisplayImage(img1, img2, title1, title2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=10)
    ax2.imshow(img2[:,:,0])
    ax2.set_title(title2, fontsize=10)

def normalized(rgb):
    if len(rgb.shape) != 3:
        raise RuntimeError('normalized: input data is not 3 dimensional data')

    if rgb.shape[2] != 3:
        raise RuntimeError('normalized: input data is not in RGB color')
        
    n_image = rgb[top:bottom,:,:]
    norm=np.zeros((n_image.shape[0], n_image.shape[1], 3), np.float32)
    norm[:,:,0]=cv2.equalizeHist(n_image[:,:,0])
    norm[:,:,1]=cv2.equalizeHist(n_image[:,:,1])
    norm[:,:,2]=cv2.equalizeHist(n_image[:,:,2])

    return norm

def preprocess_labels(label_image):

    # free up the second and third index by making them other first
    label_image[label_image[:,:,:] == 1] = 0 
    label_image[label_image[:,:,:] == 2] = 0 

    # now assign road and road line to 1
    label_image[label_image[:,:,:] == 6] = 1
    label_image[label_image[:,:,:] == 7] = 1

    # assign car to two
    label_image[label_image[:,:,:] == 10] = 2

    # the rest, for other
    label_image[label_image[:,:,:] >= 3] = 0 

    # Now, remove the hood
    # Identify all vehicle pixels
    vehicle_pixels = (label_image[:,:,0] == 2).nonzero()
    # Isolate vehicle pixels associated with the hood (y-position > 496)
    hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
    hood_pixels = (vehicle_pixels[0][hood_indices], \
                   vehicle_pixels[1][hood_indices])
    # Set hood pixel labels to 0
    label_image[hood_pixels] = 0
    return label_image[top:bottom,:,:]


def one_hot_encode(labels):
    if len(labels.shape) != 2:
        raise RuntimeError('one hot encoding: input data is not 2 dimensional data')
        
    x = np.zeros([labels.shape[0], labels.shape[1], classes])
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            x[i,j,int(labels[i][j])]=1
    return x

def prep_data(samples):
    images = []
    labels = []
    
    shuffle(samples)
    for sample in samples:
        img = cv2.cvtColor(cv2.imread ('data/Train/CameraRGB/' + sample), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(cv2.imread('data/Train/CameraSeg/' + sample), cv2.COLOR_BGR2RGB)

        images.append(normalized(img))
        label = preprocess_labels(label)
        labels.append(np.reshape(one_hot_encode(label[:,:,0]), (height*width, classes)))
        print('.',end='')        
        
    return np.array(images), np.array(labels)

def generator(samples, batch_size=6):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            labels = []
            for batch_sample in batch_samples:
                img = cv2.cvtColor(cv2.imread ('data/Train/CameraRGB/' + batch_sample), cv2.COLOR_BGR2RGB)
                label = cv2.cvtColor(cv2.imread('data/Train/CameraSeg/' + batch_sample), cv2.COLOR_BGR2RGB)

                rand = random.randrange(0,2)
                if rand == 0:
                    img = np.fliplr(img).astype(np.uint8)
                    label = np.fliplr(label).astype(np.uint8)
                
                images.append(normalized(img))
                
                label = preprocess_labels(label)
                label = one_hot_encode(label[:,:,0])
                label = np.reshape(label, (height*width, classes))
                labels.append(label)

            X_train = np.array(images)
            y_train = np.array(labels)
            yield shuffle(X_train, y_train)
            
