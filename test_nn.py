import scipy.io as spio
import numpy as np
import pandas as pd

from mat4py import loadmat
from nn import NeuronalNet

def convertYData(y):
    y = np.array(y, dtype=int)
    ret_val = np.array(np.zeros((10,len(y)), dtype=bool))
    for j in range(0,len(y)):
        ret_val[y[j], j] = True
    return ret_val

def convertXData(x):
    max_val = 255
    ret_val = np.matrix.transpose(np.array(x, dtype=float)/max_val)
    return ret_val

excel_file = 'nn_stats.xlsx'
stats = pd.DataFrame(columns=["hidden layers", "neurons per hidden layer", "activation function", "mini patch size", 
"size of training sample", "CONVERGENCE AFTER . STEPS", "PERCENT OF CORRECT PREDICTIONS"])
stats.to_excel(excel_file, index=False)

hl = [1, 2]
nphl = [700, 1000]
sots = [100, 300, 600, 6000, 60000]
af = ["sigmoid", "relu"]
mps = [5,1,10]

mat = loadmat('mnist.mat')
testX = convertXData(mat['testX'])
testY = convertYData(mat['testY'])

for h in hl:
    for n in nphl:
        for s in sots:
            trainX = convertXData(mat['trainX'])[:,0:s]
            trainY = convertYData(mat['trainY'])[:,0:s]
            net = NeuronalNet(amount_input_neurons=trainX.shape[0], amount_hidden_neurons=n,
                amount_output_neurons=trainY.shape[0], hidden_layers=h)
            for a in af:
                net.activation_function = a
                for m in mps:
                    net.mini_patch_size = m
                    tmp = net.train(trainX, trainY)
                    if tmp[0]:
                        steps=tmp[1]
                    else:
                        steps = "did not converge"
                    success = net.test(testX, testY)
                    stats = pd.read_excel(excel_file)
                    stats.loc[len(stats)] = [h,n,a,m,s,steps,success]
                    stats.to_excel(excel_file, index=False)
                    print("results saved in file", excel_file)
                    net.reset()

