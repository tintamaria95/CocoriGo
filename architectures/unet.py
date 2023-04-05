
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.layers import Input , Conv2D , MaxPooling2D , Dropout , concatenate , UpSampling2D
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
import numpy as np

planes = 31
filters = 30


def UNet(inputs):#input_shape):
  keras.backend.clear_session()

  conv1 = Conv2D(16, 2, activation = 'relu', padding = 'valid', strides=(1, 1), kernel_initializer = 'he_normal')(inputs)
  #conv1 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = Conv2D(32, 2, activation = 'relu', padding = 'valid', strides=(1, 1), kernel_initializer = 'he_normal')(pool1)
  print(conv2.shape)
  #conv2 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Conv2D(64, 2, activation = 'relu', padding = 'same', strides=(1, 1), kernel_initializer = 'he_normal')(pool2)
  #conv3 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  
  #conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  #conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  #drop4 = Dropout(0.5)(conv4)
  #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

  conv5 = Conv2D(256, 2, activation = 'relu', padding = 'same', strides=(1, 1), kernel_initializer = 'he_normal')(pool3) #4
  #conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  drop5 = Dropout(0.5)(conv5)

  #up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
  #merge6 = concatenate([drop4,up6], axis = 3)
  #conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  #conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

  up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', strides=(1, 1), kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', strides=(1, 1), kernel_initializer = 'he_normal')(merge7)
  #conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

  up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', strides=(1, 1), kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(32, 2, activation = 'relu', padding = 'same', strides=(1, 1), kernel_initializer = 'he_normal')(merge8)
  #conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

  #up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', strides=(1, 1), kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
  #merge9 = concatenate([conv1,up9], axis = 3)
  #conv9 = Conv2D(16, 2, activation = 'relu', padding = 'same', strides=(1, 1), kernel_initializer = 'he_normal')(merge9)
  #conv9 = Conv2D(2, 2, activation = 'relu', padding = 'same', strides=(1, 1), kernel_initializer = 'he_normal')(conv9)

  return conv8
  
inputs = keras.Input(shape=(19, 19, planes), name='board')
#outputs = layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)
conv9 = UNet(inputs)
policy_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(conv9)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)
value_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(conv9)
value_head = layers.Flatten()(value_head)
value_head = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0005))(value_head)

model = keras.Model(inputs=inputs, outputs=[policy_head, value_head])
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

model.save('/content/drive/MyDrive/GO_project/trained_models/unet.h5')
print('Total params:', model.count_params())
