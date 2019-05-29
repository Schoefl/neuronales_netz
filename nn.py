import numpy as np
from numpy.random import random

class NeuronalNet:
    def __init__(self, amount_input_neurons=3, amount_hidden_neurons=9, hidden_layers=1, amount_output_neurons=1):
        self.__ain = amount_input_neurons
        self.__ahn = amount_hidden_neurons
        self.__hl = hidden_layers
        self.__aon = amount_output_neurons
        self.__learning_rate = 0.001

        ## use same notation as in lecture

        ## initialize weights 
        self.__w = [None]*(self.__hl+1)
        self.__w[0] = random([self.__ahn, self.__ain]) # w for second layer
        if self.__hl > 1:
            pass # ToDo: hier noch was machen
        self.__w[len(self.__w)-1] = random([self.__aon, self.__ahn])

        ## initialize bias
        self.__b = [None]*(self.__hl+1)
        for j in range(0,self.__hl):
            self.__b[j] = random(self.__ahn)
        self.__b[len(self.__b)-1] = random(self.__aon)

        ## initialize a
        self.__a = [None]*(self.__hl+2)

        ## initialize z
        self.__z = [None]*(self.__hl+1)

        ## initialize delta
        self.__delta = [None]*(self.__hl+1)
            
    def test(self):

        x = [1,0,0.5]
        y = [0]

        count=0
        while(count < 10000):
            self.feedForward(x)
            self.backPropagation(y)
            self.gradientDescent()
            if self.mse(self.__a[len(self.__a)-1], y) < 0.00001: break
            count += 1
            # print("ROUND", count)
            # print("z \n", self.__z)
            # print("a \n", self.__a)
            # print("w \n", self.__w)
            # print("b \n", self.__b)
            # print("delta \n", self.__delta)

        print("stopped at count", count)
        print("output train: ", self.__a[len(self.__a)-1])
        self.feedForward([0,1,0])
        print("output test: ", self.__a[len(self.__a)-1])



        # self.feedForward(x)
        # self.backPropagation(y)
        # self.gradientDescent()
        # print("z \n", self.__z)
        # print("a \n", self.__a)
        # print("w \n", self.__w)
        # print("b \n", self.__b)
        # print("delta \n", self.__delta)

        #print("self.__w \n", self.__w)

    def mse(self, a, y):
        """
        input: a..output of net, y..value it should be
        output: mean sqared error 
        """
        assert(len(a)==len(y)), "a and y have different lengths"
        return np.square(np.array(a)-np.array(y)).mean()

    def sigma(self, x, deriv=False):
        ## activation function
        x = np.array(x)
        if deriv==False:
            #assert((x>=0).all() and (x<=1).all()), "input must be in range [0,1]"
            #return 1 / (1 + np.exp(-x)) # sigmoid
            ## relu
            ret_val = x
            ret_val[ret_val<0]=0
        else:
            # relu
            ret_val=np.zeros(len(x))
            ret_val[x>0]=1
        return np.array(ret_val)

    def feedForward(self, x):
        ## ToDo: effizienter gestalten
        a = self.__a 
        z = self.__z
        a[0] = np.array(x)
        for j in range(0,self.__hl+1):
            z[j] = np.matmul(self.__w[j], a[j]) + self.__b[j]
            a[j+1] = self.sigma(x=z[j])
        self.__a = a
        self.__z = z

    def backPropagation(self, y):
        assert(len(y)==len(self.__a[len(self.__a)-1])), "len(y) incorrect in backPropagation"
        delta = self.__delta
        delta[len(delta)-1] = np.multiply((self.__a[len(self.__a)-1]-y), self.sigma(self.__z[len(self.__z)-1], deriv=True))

        for j in range(len(delta)-2, -1, -1):
            delta[j] = np.multiply(np.matmul(np.matrix.transpose(self.__w[j+1]), delta[j+1]), self.sigma(self.__z[j], deriv=True))
        self.__delta = delta

    def gradientDescent(self):
        for j in range(0, len(self.__w)):
            self.__w[j] = self.__w[j]-self.__learning_rate*np.outer(self.__delta[j], np.matrix.transpose(self.__a[j]))
            self.__b[j] = self.__b[j]-self.__learning_rate*self.__delta[j]
        

### tests
nn = NeuronalNet()
nn.test()