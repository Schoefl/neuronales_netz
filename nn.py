import numpy as np
from numpy.random import random

class NeuronalNet:
    def __init__(self, amount_input_neurons=3, amount_hidden_neurons=9, hidden_layers=1, amount_output_neurons=1, mini_batch_size=3):
        self.__ain = amount_input_neurons
        self.__ahn = amount_hidden_neurons
        self.__hl = hidden_layers
        self.__aon = amount_output_neurons
        self.__learning_rate = 0.001
        self.__mbs = mini_batch_size

        ## use same notation as in lecture
        ## initialize weights 
        self.__w = [None]*(self.__hl+1)
        self.__w[0] = random([self.__ahn, self.__ain])               # w for second layer
        if self.__hl > 1:
            for j in range(1,self.__hl):                             # w for hidden layers
                self.__w[j] = random([self.__ahn, self.__ahn])
        self.__w[len(self.__w)-1] = random([self.__aon, self.__ahn]) # w for last layer


        ## initialize bias
        self.__b = [None]*(self.__hl+1)
        for j in range(0,self.__hl):
            self.__b[j] = random(self.__ahn)
        self.__b[len(self.__b)-1] = random(self.__aon)

        ## make sure self.__feedForward(x), self.__backPropagation(y) are not called in wrong context
        self.__a = None
        self.__z = None
        self.__delta = None   

    def _fit(self, x, y):
        assert(x.shape[0]==self.__ain and y.shape[0]==self.__aon and x.shape[1]==y.shape[1] and x.shape[1]!=0), "invalid input in fit"

        self.__a = [None]*(self.__hl+2)
        self.__z = [None]*(self.__hl+1)
        self.__delta = [None]*(self.__hl+1)
        delta_w = [None]*(self.__hl+1) # delta for w
        delta_b = [None]*(self.__hl+1) # delta for b

        for j in range(0, x.shape[1]):
            self.__feedForward(x[:,j])
            self.__backPropagation(y[:,j])
            for i in range(0, len(self.__delta)):
                if delta_w[i]==None:
                    delta_w[i] = np.outer(self.__delta[i], np.matrix.transpose(self.__a[i]))
                else:
                    delta_w[i] += np.outer(self.__delta[i], np.matrix.transpose(self.__a[i]))
                if delta_b[i]==None:
                    delta_b[i] = self.__delta[i]
                else:
                    delta_b[i] += self.__delta[i]

        for i in range(0, len(self.__w)):
            self.__w[i] = self.__w[i]-self.__learning_rate/x.shape[1]*delta_w[i]
            self.__b[i] = self.__b[i]-self.__learning_rate/x.shape[1]*delta_b[i]

        ret_val = self.mse(self.__a[len(self.__a)-1], y)
        ## make sure self.__feedForward(x), self.__backPropagation(y) are not called in wrong context
        self.__a = None
        self.__z = None
        self.__delta = None        
        return ret_val

    def __feedForward(self, x):
        assert(self.__a!=None and self.__z!=None), "function should not be called here"
        a = self.__a 
        z = self.__z
        a[0] = np.array(x)
        for j in range(0,self.__hl+1):
            z[j] = np.matmul(self.__w[j], a[j]) + self.__b[j]
            a[j+1] = self.sigma(x=z[j])
        self.__a = a
        self.__z = z


    def __backPropagation(self, y):
        assert(self.__a!=None and self.__z!=None and self.__delta!=None), "function should not be called here"
        assert(len(y)==len(self.__a[len(self.__a)-1])), "len(y) incorrect in __backPropagation"
        delta = self.__delta
        delta[len(delta)-1] = np.multiply((self.__a[len(self.__a)-1]-y), self.sigma(self.__z[len(self.__z)-1], deriv=True))

        for j in range(len(delta)-2, -1, -1):
            delta[j] = np.multiply(np.matmul(np.matrix.transpose(self.__w[j+1]), delta[j+1]), self.sigma(self.__z[j], deriv=True))
        self.__delta = delta


    def test(self):
        x = np.array([1,0,0.5]).reshape((3,1))
        y = np.array([0]).reshape((1,1))

        print(self._fit(x,y))


    @staticmethod
    def mse(a, y):
        """
        input: a..output of net, y..value it should be
        output: mean sqared error 
        """
        assert(len(a)==len(y)), "a and y have different lengths"
        return np.square(np.array(a)-np.array(y)).mean()

    @staticmethod
    def sigma(x, deriv=False):
        ## activation function
        x = np.array(x)
        if deriv==False:
            ## relu
            ret_val = x
            ret_val[ret_val<0]=0
        else:
            # relu
            ret_val=np.zeros(len(x))
            ret_val[x>0]=1
        return np.array(ret_val)



### tests
nn = NeuronalNet()
nn.test()