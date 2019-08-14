#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 22:25:11 2019

@author: beehive
"""

# Keras/TensorFlow Convolutional Neural Network
# Melanoma Detector
#
#---------------------
import numpy as np
#import matplotlib.pyplot as plt
import os 
import cv2
import random
import pickle
import time
import keras
#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


#-------------VARIABLES-------------#
# name of the model
t = int(time.time())
NAME = f"9-class-CNN-{t}"

#callback ot be based into the model-fit
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

# define gpu options; take a 1/3 of GPU VRAM
#gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opt))

# define your data path and categories.
DATADIR = "/home/beehive/Documents/ML-ISIC_Melanoma/Cancer_Training_Images"
CATEGORIES = ["mel_images", "nv_images", "bcc_images", "ak_images", "bkl_images", "df_images", "vasc_images", "scc_images"]

# test - define your normalized image size
IMG_SIZE = 100

X = []
y = []

# save X (feature variable) 
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

# save y (labels)
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# define feature variables and labels to be used in NN analysis
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

# test - define your normalized image size
#X = X/255.0

#-------------PLOT-------------#


#-------------FUNCTIONS-------------#
# create training data by iterating through everything and putting the data into a single array
training_data = []

def create_training_data(): 
    for category in CATEGORIES: 
        path = os.path.join(DATADIR, category) # path to mel or nv dir
        class_num = CATEGORIES.index(category) # map to a numerical value based on the list in CATEGORIES
        for img in os.listdir(path): 
            try: 
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e: 
                pass
            
#run function - create the training data
create_training_data()

#-------------NON_FUNCT-------------#
# shuffle the training data so that the neural network does not accidently force predict
#based on a predictable reoccuring data set. 
random.shuffle(training_data)

# take the shuffled data and pack into variables before NN analysis
for features, label in training_data: 
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # '-1' is for all features. '1' is for grayscale

#-------------MODEL-------------#
# create CNN model
#layer 1
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(Conv2D(32, (3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#layer 2 
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation("relu"))
model.add(Conv2D(64, (3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#layer 3 (convert 3d feature maps to 1d vector)
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#output layer, final activation
model.add(Dense(8))
model.add(Activation('softmax'))



#-------------TRAIN-------------#
# set RMSprop optimizer 
#rmsp_opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# set Stochastic Gradient Descent
#sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9 nesterov=True)

# Define the parameters for the training of the model
#optimizer is used to minimize loss, loss... , metrics is how the model is perceived.
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer="adam", 
              metrics=['accuracy'])

#train the model; to see training in terminal type: "tensorboard --logdir=logs/"
model.fit(X, y, batch_size=32, epochs = 1, validation_split=0.1, callbacks=[tensorboard])

#-------------LOSS/ACCURACY-------------#

#-------------INITIATE-------------#
if __name__ == '__main__': 
    #pass
    print("End CNN Model")
