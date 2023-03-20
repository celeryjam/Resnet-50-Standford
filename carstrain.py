# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:49:31 2023

@author: Jack Pham
"""
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import math
import random
import functools
import numpy as np
import pandas as pd
import csv
import os
import cv2

img_folder = 'cars_train'
IMG_WIDTH=128
IMG_HEIGHT=128

annots = loadmat('cars_train_annos.mat')
metas = loadmat('cars_meta.mat')

data = [[row.flat[0] for row in line] for line in annots['annotations'][0]]
columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class','fname']
df_train = pd.DataFrame(data, columns=columns)
df_train = df_train.set_index('fname')


plt.figure(figsize=(20,20))

'''
test reading image

for i in range(5):
    file = random.choice(os.listdir(img_folder))
    image_path= os.path.join(img_folder, file)
    img=mpimg.imread(image_path)
    ax=plt.subplot(1,5,i+1)
    ax.title.set_text(file)
    plt.imshow(img)
  '''
'''
'''
def create_dataset(img_folder, limit=8144):
    img_data=[]
    class_name=[]
    for dir1 in os.listdir(img_folder):
        if limit>0:
            image_path=os.path.join(img_folder,dir1)
            #print(image_path)
            image=cv2.imread(image_path, cv2.IMREAD_COLOR )
            image=image[df_train['bbox_y1'].loc[dir1]:df_train['bbox_y2'].loc[dir1],df_train['bbox_x1'].loc[dir1]:df_train['bbox_x2'].loc[dir1]]
            #plt.imshow(image)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image=image.astype('float32')
            image/=255
            img_data.append(image)
            class_name.append(df_train['class'].loc[dir1])
            limit-=1
    class_name = keras.utils.to_categorical(class_name)
    img_data=np.array(img_data)
    return img_data, class_name

# extract the image array and class name
img_data, class_name =create_dataset(img_folder)
input_shape = img_data[0].shape

  
def build_resnet_model():
    model=Sequential()
    res_model = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=197
    )
    model.add(res_model)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(197, activation='softmax'))
    opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X,Y,build_model,folds=10,epochs=32,batch_size=4):
    model=build_model()
    kf = KFold(n_splits=folds)
    fold=0
    for train_index, test_index in kf.split(X):
        fold += 1
        print("Starting fold ", fold)
        print("===========================================")
        X_train = np.array(X[train_index])
        Y_train = np.array(Y[train_index])
        X_test = np.array(X[test_index])
        Y_test = np.array(Y[test_index])  
        print("Starting training ")
        es = EarlyStopping(monitor='train_acc', mode='min', verbose=1)
        train_hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)
        print("Evaluating with ",len(X_test), " test cases")
        _, acc = model.evaluate(X_test, Y_test, verbose=0)
        print("Model Accuracy: {:5.2f}%".format(100*acc))

model = build_resnet_model()        
train_hist = model.fit(img_data, class_name, batch_size=4, epochs=64, verbose=1)
        