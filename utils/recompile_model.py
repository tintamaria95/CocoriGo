import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras import regularizers

MODEL = "shufflenet.h5"
model_name = "/content/drive/MyDrive/GO_project/trained_models/" + MODEL
model = keras.models.load_model(model_name)

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.00005, momentum=0.9),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

model.save('/content/drive/MyDrive/GO_project/trained_models/' + MODEL)