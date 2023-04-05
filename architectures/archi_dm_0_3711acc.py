import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt


planes = 31
filters = 31

input = keras.Input(shape=(19, 19, planes), name='board')

#Block 1
x_1 = layers.Conv2D(filters, 3, strides=(1, 1), padding='same')(input)
b_1 = layers.BatchNormalization()(x_1)
r_1 = layers.ReLU()(b_1)
x_2 = layers.Conv2D(filters, 3, strides=(1, 1), padding='same')(r_1)
b_1_bis = layers.BatchNormalization()(x_2)
z_1 = tf.keras.layers.Add()([b_1_bis, input])
z_1_bis = layers.ReLU()(z_1)

#Block 2
x_3 = layers.Conv2D(filters, 3, strides=(1, 1), padding='same')(z_1_bis)
b_2 = layers.BatchNormalization()(x_3)
r_2 = layers.ReLU()(b_2)
x_4 = layers.Conv2D(filters, 3, strides=(1, 1), padding='same')(r_2)
b_2_bis = layers.BatchNormalization()(x_4)
z_2 = tf.keras.layers.Add()([b_2_bis, z_1_bis])
z_2_bis = layers.ReLU()(z_2)

#Block 3
x_5 = layers.Conv2D(filters, 3, strides=(1, 1), padding='same')(z_2_bis)
b_3 = layers.BatchNormalization()(x_5)
r_3 = layers.ReLU()(b_3)
x_6 = layers.Conv2D(filters, 3, strides=(1, 1), padding='same')(r_3)
b_3_bis = layers.BatchNormalization()(x_6)
z_3 = tf.keras.layers.Add()([b_3_bis, z_2_bis])
z_3_bis = layers.ReLU()(z_3)

#Block 4
x_7 = layers.Conv2D(filters, 3, strides=(1, 1), padding='same')(z_3_bis)
b_4 = layers.BatchNormalization()(x_7)
r_4 = layers.ReLU()(b_4)
x_8 = layers.Conv2D(filters, 3, strides=(1, 1), padding='same')(r_4)
b_4_bis = layers.BatchNormalization()(x_8)
z_4 = tf.keras.layers.Add()([b_4_bis, z_3_bis])
z_4_bis = layers.ReLU()(z_4)

#Final
x_9 = layers.Conv2D(filters, 3, strides=(1, 1), padding='same')(z_4_bis)
b_5 = layers.BatchNormalization()(x_9)
r_5 = layers.ReLU()(b_5)

policy_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(r_5)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)
value_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(r_5)
value_head = layers.Flatten()(value_head)
value_head = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)

model = keras.Model(inputs=input, outputs=[policy_head, value_head])

model.summary ()

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.003, momentum=.9),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})
model.save ('test.h5')
#plt.show(keras.utils.plot_model(model, str('test_0' + ".png"), show_shapes=True))