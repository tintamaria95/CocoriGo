import sys
sys.path.insert(0,'/content')
from tensorflow import keras
import numpy as np
import gc
import os

import golois

planes = 31
moves = 361
N = 10000
batch = 128

input_data = np.random.randint(2, size=(N, 19, 19, planes))
input_data = input_data.astype ('float32')
policy = np.random.randint(moves, size=(N,))
policy = keras.utils.to_categorical (policy)
value = np.random.randint(2, size=(N,))
value = value.astype ('float32')
end = np.random.randint(2, size=(N, 19, 19, 2))
end = end.astype ('float32')
groups = np.zeros((N, 19, 19, 1))
groups = groups.astype ('float32')
#print ("getValidation", flush = True)
golois.getValidation (input_data, policy, value, end)
golois.getValidation (input_data, policy, value, end)

# Load models
h5_files = []
for name in os.listdir('/content/drive/MyDrive/GO_project/trained_models'):
  if name[-2:] == 'h5':
    h5_files.append(name)

# if only one model
#h5_files = ['model_resnet_Adam_lr1_10-4.h5']
for model_filename in h5_files:
  model = keras.models.load_model(
    '/content/drive/MyDrive/GO_project/trained_models/' + 
    model_filename)
  m_names = model.metrics_names

  val = model.evaluate (input_data,
                        [policy, value], verbose = 0, batch_size=batch)
  print("")
  print("*******************")
  print(model_filename)
  print('Total params:', model.count_params())
  for i, metric in enumerate(m_names):
    print(metric, ":", val[i])
# display(keras.utils.plot_model(model, str(model_name + ".png"), show_shapes=True))