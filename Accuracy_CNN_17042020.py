# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:48:27 2020

@author: Hira
"""

import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Add, Input, Multiply, Concatenate, GlobalAveragePooling2D
#from keras.applications.vgg16 import VGG16
from keras.applications.mobilenetv2 import MobileNetV2
import os, glob, random
from PIL import Image
import pickle
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

batch_size = 11
num_classes = 2
#epochs = 50
epochs = 2
#data_augmentation = False
#num_predictions = 20
#save_dir = os.path.join(os.getcwd(), 'saved_models')
#model_path = 'C:/Users/Hira/Desktop/Coffee Bean Size_05032020/801010_format/32_32/Coffee_2019/mbnv2_model_5.h5'
model_path = 'C:/Users/Hira/Desktop/Coffee Bean Size_05032020/801010_format/imagedata_01/256_256/Coffee_2019/mbnv2_model_5.h5'

img_width = 128
img_height = 128

train_dir = 'C:/Users/Hira/Desktop/Coffee Bean Size_05032020/801010_format/imagedata_01/256_256/Coffee_2019/Train/'
validation_dir = 'C:/Users/Hira/Desktop/Coffee Bean Size_05032020/801010_format/imagedata_01/256_256/Coffee_2019/Validation/'

class_label = ['OK', 'NG']

#input_tensorの定義
input_tensor = Input(shape=(img_width, img_height, 3))
#Create the base model from the pre-trained convnets
model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor)

#model.summary()

top_model = Sequential()
#top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(GlobalAveragePooling2D())  #Global Average Pooling 層の良いポイント,パラメーター数を非常に少なくすることができる→　モデルが単純になり、過学習をしにくくなる
top_model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
top_model.add(BatchNormalization())
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# mobilenetv2とtop_modelを連結
model = Model(inputs=model.input, outputs=top_model(model.output))

model.summary()
from keras.callbacks import ModelCheckpoint

#img_width, img_height = 128, 128
nb_train_samples = 2003
nb_validation_samples = 1001 
epochs = 1
batch_size = 11
nb_category = 2

callbacks = list()
#callbacks.append(ModelCheckpoint(filepath="C:/Users/Hira/Desktop/Coffee Bean Size_05032020/801010_format/32_32/Coffee_2019/mbnv2_model_5.h5"))
callbacks.append(ModelCheckpoint(filepath="C:/Users/Hira/Desktop/Coffee Bean Size_05032020/801010_format/imagedata_01/256_256/Coffee_2019/mbnv2_model_5.h5"))

model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=0.0001),
                  metrics=['accuracy'])


check_point = keras.callbacks.ModelCheckpoint(
        filepath = os.path.join('mbnv2_model{epoch:02d}-vacc{val_acc:.2f}.hdf5'), 
        monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    

train_datagen=ImageDataGenerator(
    rescale=1./255,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True
)

validation_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='binary',
    #shuffle=True
)

validation_generator=validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='binary',
    #shuffle=True
)

history = model.fit_generator(
        train_generator,
        steps_per_epoch=25,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=10,
        #callbacks=[check_point]
        callbacks=callbacks
)

model.save(model_path)
print('\nSaved trained model at --> %s ' % model_path)

from keras.optimizers import SGD,Adam
#for test
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import os,random
from keras.preprocessing.image import img_to_array, load_img
import time

print("Wainting...")
timestamp1 = time.time()

#img_width, img_height = 128,128
nb_test_samples = 2
#batch_size = 1
nb_category = 2

batch_size=11
#file_name='vgg16_been_224'
test_dir='C:/Users/Hira/Desktop/Coffee Bean Size_05032020/801010_format/imagedata_01/256_256/Coffee_2019/Test/'
#display_dir='/home/reeen/Documents/keras/cnn3display'
label=['OK','NG']

print("Wainting...")

#load model and weights
#json_string=open(file_name+'.json').read()
#model=model_from_json(json_string)
model.load_weights(model_path)
#model.load_weights('/home/reeen/Documents/keras/cnn5/models/vgg16_weight_epoch978.h5')

model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#data generate
test_datagen=ImageDataGenerator(rescale=1./255)

test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='binary',
    #shuffle=True
)

#evaluate model
score=model.evaluate_generator(
    test_generator,
    steps=nb_test_samples,
     verbose=1)

timestamp2 = time.time()
print('\n test loss:',score[0])
print('\n test_acc:',score[1])
print("This took {} seconds for one image".format(timestamp2 - timestamp1))
