import scipy.io as spio
import numpy as np

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
    ret_val = np.matrix.transpose((np.asarray(mat['trainX'])/max_val))
    return ret_val


mat = loadmat('mnist.mat')
# test = np.zeros((784,2))
# test = np.matrix.transpose((np.asarray(mat['trainX'])/255))[:,0:1]
# print(test[test!=0])
# mat = spio.loadmat('mnist.mat', squeeze_me=False)#, mat_dtype="double")

#trainX = np.matrix.transpose((np.asarray(mat['trainX'])/255))[:,1:4]
trainX = convertXData(mat['trainX'])[:,0:10]
trainY = convertYData(mat['trainY'])[:,0:10]

# print(trainX.shape(), trainY.shape())
# tmp1 = trainX[:,0]
# tmp2 = trainX[:,2]
# print(tmp1[tmp1!=0])
# print(tmp2[tmp2!=0])
# print(all(trainX[:,0]==trainX[:,1]))
# print(all(trainY[:,0]==trainY[:,2]))


net = NeuronalNet(amount_input_neurons=trainX.shape[0], amount_hidden_neurons=round(trainX.shape[0]), amount_output_neurons=trainY.shape[0])
net.train(trainX, trainY)
#net.test(trainX)