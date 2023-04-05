import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras import regularizers

planes = 31
filters = 25


def block(input: keras.layers, nb_filters):
  x_1 = layers.Conv2D(filters, 3, strides=(1, 1), padding='same')(input)
  b_1 = layers.BatchNormalization()(x_1)
  r_1 = layers.ReLU()(b_1)
  x_2 = layers.Conv2D(filters, 5, strides=(1, 1), padding='same')(r_1)
  b_1_bis = layers.BatchNormalization()(x_2)

  i_1 = layers.Conv2D(filters, 1, strides=(1, 1), padding='same')(input)
  n_1 = layers.BatchNormalization()(i_1)

  z_1 = tf.keras.layers.Add()([b_1_bis, n_1])
  z_1_bis = layers.ReLU()(z_1)
  return z_1_bis

input_model = keras.Input(shape=(19, 19, planes), name='board')
x = block(input=input_model, nb_filters=filters)
for i in range(3):
  x = block(x, nb_filters=filters)
policy_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)
value_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
value_head = layers.Flatten()(value_head)
value_head = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)
model = keras.Model(inputs=input_model, outputs=[policy_head, value_head])

model.summary ()

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.003, momentum=.9),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

#display(keras.utils.plot_model(model, str('test_0' + ".png"), show_shapes=True))
model.save ('test.h5')