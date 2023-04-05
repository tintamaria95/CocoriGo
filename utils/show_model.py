import sys
sys.path.insert(0,'/content')
import matplotlib.pyplot as plt
import tensorflow.keras as keras

path_model = "/content/drive/MyDrive/GO_project/trained_models/"
path_img = "/content/drive/MyDrive/GO_project/archiImages/"
model_name = "shufflenet.h5"
model = keras.models.load_model(path_model + model_name)
keras.utils.plot_model(model, (path_img + model_name + ".png"), show_shapes=True)

