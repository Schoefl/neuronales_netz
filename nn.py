import numpy as np
from numpy.random import random

class NeuronalNet:
    def __init__(self, amount_input_neurons=3, amount_hidden_neurons=9, hidden_layers=1, amount_output_neurons=1, mini_batch_size=3):
        self.__ain = amount_input_neurons
        self.__ahn = amount_hidden_neurons
        self.__hl = hidden_layers
        self.__aon = amount_output_neurons

        self.learning_rate = 0.001
        self.mbs = mini_batch_size
        self.error_th = 0.001
        random_multi_w = 0.0001
        random_multi_b = 0.000000001
        add = 1/2

        ## use same notation as in lecture
        ## initialize weights 
        self.__w = [None]*(self.__hl+1)
        self.__w[0] = (random([self.__ahn, self.__ain])-add)*random_multi_w            # w for second layer
        if self.__hl > 1:
            for j in range(1,self.__hl):                             # w for hidden layers
                self.__w[j] = (random([self.__ahn, self.__ahn])-add)*random_multi_w
        self.__w[len(self.__w)-1] = (random([self.__aon, self.__ahn])-add)*random_multi_w # w for last layer


        ## initialize bias
        self.__b = [None]*(self.__hl+1)
        for j in range(0,self.__hl):
            self.__b[j] = (random(self.__ahn)-add)*random_multi_b
        self.__b[len(self.__b)-1] = (random(self.__aon)-add)*random_multi_b

        ## make sure self.__feedForward(x), self.__backPropagation(y) are not called in wrong context
        self.__a = None
        self.__z = None
        self.__delta = None   
    
    def train(self, x, y, max_steps=10000):
        ## note that numpy arrays are passed by reference, so passing x,y as often as I do it here is not inefficient
        assert(x.shape[0]==self.__ain and y.shape[0]==self.__aon and x.shape[1]==y.shape[1] and x.shape[1]!=0), "invalid input in fit"
        step = 0
        while self._fit(x,y):
            if step>=max_steps: break
            step += 1
        print("stopped after {} steps".format(step))

    def _fit(self, x, y):
        assert(x.shape[0]==self.__ain and y.shape[0]==self.__aon and x.shape[1]==y.shape[1] and x.shape[1]!=0), "invalid input in fit"

        self.__a = [None]*(self.__hl+2)
        self.__z = [None]*(self.__hl+1)
        self.__delta = [None]*(self.__hl+1)
        delta_w = [None]*(self.__hl+1) # delta for w
        delta_b = [None]*(self.__hl+1) # delta for b
        mean_mse = 0
        ret_val = True

        for j in range(0, x.shape[1]):
            self.__a = [None]*(self.__hl+2)
            self.__feedForward(x[:,j])
            self.__backPropagation(y[:,j])
            mean_mse += self.mse(self.__a[len(self.__a)-1], y[:,j])
            for i in range(0, len(self.__delta)):
                if type(delta_w[i])==type(None):
                    delta_w[i] = np.outer(self.__delta[i], np.matrix.transpose(self.__a[i]))
                else:
                    delta_w[i] += np.outer(self.__delta[i], np.matrix.transpose(self.__a[i]))
                if type(delta_b[i])==type(None):
                    delta_b[i] = self.__delta[i]
                else:
                    delta_b[i] += self.__delta[i]
            # print("y: ", y[:,j])
            # print("output: ", self.__a[len(self.__a)-1])
        print("mse", mean_mse/(j+1), "y: ", np.argmax(y[:,j])+1,"output: ", np.argmax(self.__a[len(self.__a)-1])+1)
        mean_mse = mean_mse/x.shape[1]
        # print("mse", mean_mse,"y: ", y[:,j])
        # print("mse", mean_mse, "output: ", self.__a[len(self.__a)-1], np.argmax(self.__a[len(self.__a)-1])+1)
        #print("mse", mean_mse)
        if mean_mse > self.error_th:
            for i in range(0, len(self.__w)):
                self.__w[i] = self.__w[i]-self.learning_rate/x.shape[1]*delta_w[i]
                self.__b[i] = self.__b[i]-self.learning_rate/x.shape[1]*delta_b[i]
        else: 
            ret_val=False

        ## make sure self.__feedForward(x), self.__backPropagation(y) are not called in wrong context
        self.__a = None
        self.__z = None
        self.__delta = None        
        return ret_val

    def __feedForward(self, x):
        assert(self.__a!=None and self.__z!=None), "function should not be called here"
        ## assign new names to make clear which values are going to be changed, really this is just a reference
        ## to the same variable, so if a gets changed, self.__a is changed as well
        #a = self.__a 
        #z = self.__z
        self.__a[0] = np.array(x)
        # print("self.__a[0].shape", self.__a[0].shape)
        for j in range(0,self.__hl+1):
            # print("self.__w[{}].shape".format(j), self.__w[j].shape)
            self.__z[j] = np.inner(self.__w[j],self.__a[j]) + self.__b[j]
            self.__a[j+1] = self.sigma(x=self.__z[j], deriv=False)
        #     print("self.__a[{}].shape".format(j+1), self.__a[j+1].shape)
        # print("self.__z[0]", self.__z[0])
        # print("self.__z[1]", self.__z[1])
        # print("self.__a[1]", self.__a[1])
        # print("self.__a[2]", self.__a[2])

        # print(self.__w[j].shape, self.__a[j].shape)
        # print("np.matmul(self.__w[j], self.__a[j])",np.outer(self.__w[j], self.__a[j]))
        # print("self.__a[j-1]", self.__a[j-1])
        # print("self.__w[j]", self.__w[j])

    def __backPropagation(self, y):
        assert(self.__a!=None and self.__z!=None and self.__delta!=None), "function should not be called here"
        assert(len(y)==len(self.__a[len(self.__a)-1])), "len(y) incorrect in __backPropagation"
        delta = self.__delta # create new reference to self.__delta same as in __feedForward
        self.__delta[len(self.__delta)-1] = np.multiply((self.__a[len(self.__a)-1]-y), self.sigma(self.__z[len(self.__z)-1], deriv=True))

        for j in range(len(self.__delta)-2, -1, -1):
            self.__delta[j] = np.multiply(np.matmul(np.matrix.transpose(self.__w[j+1]), self.__delta[j+1]), self.sigma(self.__z[j], deriv=True))


    def test(self, x):
        self.__a = [None]*(self.__hl+2)
        self.__z = [None]*(self.__hl+1)
        self.__feedForward(x[:,0])
        #print("self.__a[len(self.__a)-1]", self.__a[len(self.__a)-1])
        self.__feedForward(x[:,1])
        #print("self.__a[len(self.__a)-1]", self.__a[len(self.__a)-1])


    @staticmethod
    def mse(a, y):
        """
        input: a..output of net, y..value it should be
        output: mean sqared error 
        """
        assert(len(a)==len(y)), "a and y have different lengths"
        return np.square(np.array(a)-np.array(y)).mean()

    @staticmethod
    def sigma(x, deriv=False, func="sigmoid"):
        ## activation function
        #x = np.array(x)
        if func=="sigmoid":
            if deriv==False:
                ## relu
                # ret_val = x
                # ret_val[ret_val<0]=0
                ret_val = 1/(1+np.exp(-x))
                #print("ret_val", ret_val)
            else:
                ## relu
                # ret_val=np.zeros(len(x))
                # ret_val[x>0]=1
                ret_val = NeuronalNet.sigma(x, deriv=False)*(1-NeuronalNet.sigma(x, deriv=False))
        elif func=="relu":
            if deriv==False:
                ret_val = x
                ret_val[ret_val<0]=0
            else:
                ret_val=np.zeros(len(x))
                ret_val[x>0]=1
        else:
            assert(False), "Invalide Function"
        #ret_val = 1/(1+np.exp(-x))
        return ret_val
