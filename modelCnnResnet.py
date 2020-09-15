import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import tensorflow as tf
import h5py
import keras
from keras import Model, metrics
from keras.layers import Input,Conv2D,BatchNormalization,Activation, concatenate,UpSampling2D, Dense
from keras.applications import ResNet50
from matplotlib import pyplot as plt

def UnProjection(x):
    x_2 = UpSampling2D()(x)
    x_2 = Conv2D(128,(5,5),padding='same')(x_2)
    x_2 = UpSampling2D()(x_2)
    x = UpSampling2D()(x)
    x = Conv2D(128,(5,5),padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128,(3,3),padding='same')(x)
    x = UpSampling2D()(x)
    x = concatenate([x_2,x])
    return x

hf = h5py.File('dataset.h5','r')
images = hf.get('images')
depths = hf.get('depths')

x_train,x_test,y_train,y_test = train_test_split(images,depths,test_size=0.33,shuffle=False)

input = Input(shape=(480,640,3))
x = ResNet50(weights='imagenet',include_top = False)(input)
x = Conv2D(128,(1,1),padding='same')(x)
x = BatchNormalization()(x)
x = UnProjection(x)
x = UpSampling2D()(x)
x = UpSampling2D()(x)
x = UpSampling2D()(x)
output = Conv2D(256,(3,3),padding='same',activation='relu')(x)
model = Model(inputs = input,outputs = output)
model.summary()
model.compile(optimizer = 'adam',loss='mse',metrics = ['mse'])
# y_pred = model.predict(x_test,10)
history = model.fit(x_train, y_train,validation_split = 0.1, epochs=8, batch_size=1)
score, accu = model.evaluate(x_test,y_test)
print('Test Score:',score)
print('Test accuracy:',accu)
#model accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# model.fit(x_train,y_train,batch_size = 100,epochs = 15,verbose = 1,validation_data=(x_test,y_test))
#loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('losss')
plt.xlabel('epoch')
plt.legend('train',loc='upper left')
plt.show()
