import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras import regularizers

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense
from tensorflow.keras.models import Model

planes = 31
filters = 31

def resnet_block(inputs, num_filters, kernel_size, strides):
    x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, inputs])
    x = Activation('relu')(x)
    return x

inputs = keras.Input(shape=(19, 19, planes), name='board')

x = Conv2D(filters, kernel_size=3, strides=(1, 1), padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = resnet_block(x, filters, kernel_size=3, strides=(1, 1))
x = resnet_block(x, filters, kernel_size=3, strides=(1, 1))
x = resnet_block(x, filters, kernel_size=3, strides=(1, 1))
x = resnet_block(x, filters, kernel_size=3, strides=(1, 1))

#x = Flatten()(x)
#x = Dense(50, activation='softmax')(x)

policy_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)
value_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
value_head = layers.Flatten()(value_head)
value_head = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0005))(value_head)

model = keras.Model(inputs=inputs, outputs=[policy_head, value_head])

model.summary ()

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.005),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

model.save('/content/drive/MyDrive/GO_project/trained_models/resnet_SGD.h5')
print('Total params:', model.count_params())