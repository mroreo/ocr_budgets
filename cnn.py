# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:20:38 2019

@author: bcheung
"""
import cv2
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils, Sequence
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

CHARS ='AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789.:,;"(!?)+-*/='

targetLE = LabelEncoder()
targetLE.fit(np.array([c for c in CHARS]))

batch_size = 128
num_classes = len(CHARS)

IMG_ROWS, IMG_COLS = 64, 64

train_labels = pd.read_csv('./train_labels.csv')
valid_labels = train_labels.sample(int(0.3*len(train_labels)))
train_labels = train_labels[~train_labels['id_key'].isin(valid_labels['id_key'])]

test_labels = pd.read_csv('./test_labels.csv')

def loadFiles(jpg_list,directory):
    
    jpgs = []
    for jpg_file in jpg_list:
        image_path = '{}/{}.jpg'.format(directory,jpg_file)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        jpgs.append(gray)
    return(np.array(jpgs).reshape(-1,64,64,1))

def imageLoader(x,targetLE,y=None,directory='./train_labels',batch_size=100):
    
    L = len(train_labels)
    
    while True:
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            limit = min(batch_end,L)
            X = loadFiles(x[batch_start:limit],directory)
        
            Y = targetLE.transform(y[batch_start:limit])
            Y = np_utils.to_categorical(Y,len(CHARS))
            
            batch_start += batch_size
            batch_end += batch_size
            
            yield(X,Y)
            
class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, list_IDs, labels, labelEncoder, directory, data_augmentation=None,batch_size=32, dim=(64,64), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels
        self.labelEncoder = labelEncoder
        self.directory = directory
        self.data_augmentation = data_augmentation
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def load_data(self,jpg_list):
        
        jpgs = []
        for jpg_file in jpg_list:
            image_path = '{}/{}.jpg'.format(self.directory,jpg_file)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            jpgs.append(gray)
        return(np.array(jpgs).reshape(-1,*self.dim,self.n_channels))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return(X, y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, idx):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = self.load_data(self.list_IDs[idx])
         # Data Augmentation
        if not(self.data_augmentation is None):
            X = np.array([self.data_augmentation.random_transform(x) for x in X])
        
        if self.labels is None:
            Y = None
        else:
            Y = self.labelEncoder.transform(self.labels[idx])
            Y = np_utils.to_categorical(Y,self.n_classes)

        return(X, Y)
            
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.6,
    height_shift_range=0.6,
    zoom_range=0.6)

train_generator = DataGenerator(list_IDs=train_labels['id_key'].values, 
                                labels=train_labels['target'].values, 
                                labelEncoder=targetLE, 
                                directory='./train_labels', 
                                batch_size=512, 
                                data_augmentation=datagen,
                                dim=(64,64), 
                                n_channels=1,
                                n_classes=num_classes, 
                                shuffle=True)

valid_generator = DataGenerator(list_IDs=valid_labels['id_key'].values, 
                                labels=valid_labels['target'].values, 
                                labelEncoder=targetLE, 
                                directory='./train_labels', 
                                batch_size=512, 
                                data_augmentation=datagen,
                                dim=(64,64), 
                                n_channels=1,
                                n_classes=num_classes, 
                                shuffle=True)

test_generator = DataGenerator(list_IDs=test_labels['id_key'].values, 
                                labels=None, 
                                labelEncoder=targetLE, 
                                directory='./test_labels', 
                                batch_size=512, 
                                dim=(64,64), 
                                n_channels=1,
                                n_classes=num_classes, 
                                shuffle=False)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(IMG_ROWS,IMG_ROWS,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit_generator(generator=train_generator,
                    validation_data=valid_generator,
                    epochs=10000)

test_probs = model.predict_generator(test_generator)
pred_labels = np.argmax(test_probs,axis=1)
test_labels_subset = test_labels.iloc[:1028]
test_labels_subset['pred_label'] = targetLE.inverse_transform(pred_labels)
            