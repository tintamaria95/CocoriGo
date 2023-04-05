import sys
sys.path.insert(0,'/content')
import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras import regularizers
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import golois

MODEL = "shufflenet.h5"

def main():

    epochs = 100
    epochVal = 10

    planes = 31
    moves = 361
    N = 10000
    batch = 32

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
    golois.getValidation (input_data, policy, value, end)
    

    # Load models
    model_name = "/content/drive/MyDrive/GO_project/trained_models/" + MODEL
    model = keras.models.load_model(model_name)
    m_names = model.metrics_names

    path_res = "/content/drive/MyDrive/GO_project/results_training/"
    f = open(path_res + MODEL.split('.')[0] + ".txt", 'a')
    f.close()
    f = open(path_res + MODEL.split('.')[0] + ".txt", 'r')
 
    if(os.stat(path_res + MODEL.split('.')[0] + ".txt").st_size != 0):
      lines = f.readlines()
      iters_str = lines[0][:-1] + ','  # Remove back to line
      res_mse_str = lines[1][:-1] + ','
      res_pol_acc_str = lines[2][:-1] + ','
    else:
      iters_str = ''
      res_mse_str = ''
      res_pol_acc_str = ''
    f.close()
    
    iters = []
    res_mse = []
    res_policy_acc = []

    ####################
    for i in tqdm(range(1, epochs + 1)):
        # print ('epoch ' + str (i))
        golois.getBatch (input_data, policy, value, end, groups, i * N)
        history = model.fit(input_data,
                            {'policy': policy, 'value': value}, 
                            epochs=1, batch_size=batch, verbose=0)
        if (i % epochs == 0 or i % epochVal == 0):
            gc.collect ()
            golois.getValidation (input_data, policy, value, end)
            val = model.evaluate (input_data,
                                [policy, value], verbose = 0, batch_size=batch)
            print ("policy_accuracy =", val[3])
            # save results
            iters.append(i)
            res_mse.append(val[4])
            res_policy_acc.append(val[3])
            model.save (model_name)
    ################

    # Dans le cas où on réentraîne, on ne repart pas d'une iter=0
    prevIters = 0
    if(iters_str != ''):
      prevIters = int(iters_str.split(',')[-2])
    for i in range(len(iters)):
        iters_str += (str(iters[i] + prevIters) + ',')
        res_mse_str += (str(res_mse[i]) + ',')
        res_pol_acc_str += (str(res_policy_acc[i]) + ',')
    # delete last ',' and back to line (csv format)
    iters_str = iters_str[: -1] + '\n'
    res_mse_str = res_mse_str[: -1]  + '\n'
    res_pol_acc_str = res_pol_acc_str[: -1]  + '\n'
    # Add lines to txt file
    f = open(path_res + MODEL.split('.')[0] + ".txt", 'w')
    f.writelines(iters_str)
    f.writelines(res_mse_str)
    f.writelines(res_pol_acc_str)

    print("")
    print("*******************")
    print(model_name)
    for i, metric in enumerate(m_names):
        print(metric, ":", val[i])
    
    
if __name__ == "__main__":
    main()

