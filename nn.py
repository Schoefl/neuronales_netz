import numpy as np
from numpy.random import random

class NeuronalNet:
    """
    neuronal net to classify handwritten digits (in theorie it could be used for other things as well, I guess);
    notation is the same as in lecture
    """
    def __init__(self, amount_input_neurons, amount_hidden_neurons, amount_output_neurons, hidden_layers=1):
        ## unchangable parameters, private-like (not really, just "protected" by name mangeling, but should not be changed)
        self.__ain = amount_input_neurons
        self.__ahn = amount_hidden_neurons
        self.__hl = hidden_layers
        self.__aon = amount_output_neurons

        ## changeable parameters, public-like
        self.activation_function = "sigmoid"             
        self.mini_patch_size = 1        
        self.success_percentage = 0.98 # what percentage of training samples must be classified correctly
        self.mse_th = 0.0001           # minimum mean squared error to change weights and biases

        ## initialize weights, biases and dynamically changing parameters
        self.reset()       

    def __feedForward(self, x):
        """
        input:  numeric input vector of length amount_input_neurons (self.__ain)
        output: none
        note:   self.__a and self.__z must be initialized with right size (not None); private-like function;
        """
        assert(self.__a!=None and self.__z!=None) , "function should not be called here"
        assert(len(x)==self.__ain), "len(x) incorrect in __feedForward"
        self.__a[0] = np.array(x)
        for j in range(0,self.__hl+1):
            self.__z[j] = np.inner(self.__w[j],self.__a[j]) + self.__b[j]
            self.__a[j+1] = self.sigma(x=self.__z[j], deriv=False, func=self.activation_function)

    def __backPropagation(self, y):
        """
        input:  numeric input vector of length amount_output_neuron (self.__aon)
        output: none
        note:   self.__a, self.__delta and self.__z must be initialized with right size (not None); private-like function;
        """
        assert(self.__a!=None and self.__z!=None and self.__delta!=None), "function should not be called here"
        assert(len(y)==len(self.__a[len(self.__a)-1])), "len(y) incorrect in __backPropagation"
        ## calculate delta according to lecture notes
        self.__delta[len(self.__delta)-1] = np.multiply((self.__a[len(self.__a)-1]-y), 
            self.sigma(self.__z[len(self.__z)-1], deriv=True, func=self.activation_function))
        for j in range(len(self.__delta)-2, -1, -1):
            self.__delta[j] = np.multiply(np.matmul(np.matrix.transpose(self.__w[j+1]), self.__delta[j+1]), 
                self.sigma(self.__z[j], deriv=True))
   
    def _fit(self, x, y):
        """
        input:
          - x: numpy array with dimensions (amount_input_neurons, mini_patch_size), input for net
          - y: numpy array with dimensions (amount_output_neurons, mini_patch_size), desired output
        output: percentage of correctly classified digits
        """
        assert(x.shape[0]==self.__ain and y.shape[0]==self.__aon and x.shape[1]==y.shape[1] and x.shape[1]!=0), "invalid input in fit"
        
        ## variables needed by __feedForward and __backPropagation
        self.__a = [None]*(self.__hl+2)
        self.__z = [None]*(self.__hl+1)
        self.__delta = [None]*(self.__hl+1)

        delta_w = [None]*(self.__hl+1) # delta for w
        delta_b = [None]*(self.__hl+1) # delta for b
        mean_mse = 0      # variable to calculate mean of mean squared errors
        success_count = 0 # to count how often the prediction if a sample digit was correct

        ## iterate through amount of samples to fit 
        for j in range(0, x.shape[1]):

            self.__feedForward(x[:,j])
            self.__backPropagation(y[:,j])

            ## add up mean squared errors
            mean_mse += self.mse(self.__a[len(self.__a)-1], y[:,j])

            ## add up changes for w and b
            for i in range(0, len(self.__delta)):
                if type(delta_w[i])==type(None):
                    delta_w[i] = np.outer(self.__delta[i], np.matrix.transpose(self.__a[i]))
                else:
                    delta_w[i] += np.outer(self.__delta[i], np.matrix.transpose(self.__a[i]))
                if type(delta_b[i])==type(None):
                    delta_b[i] = self.__delta[i]
                else:
                    delta_b[i] += self.__delta[i]

            ## count amount of right predictions
            success_count+=int(np.argmax(y[:,j])==np.argmax(self.__a[len(self.__a)-1]))

        ## if mean of errors is greater than error threshold (mse_th), update weights and biases
        if mean_mse/x.shape[1] > self.mse_th:
            for i in range(0, len(self.__w)):
                self.__w[i] -= self.__learning_rate*delta_w[i]/x.shape[1]
                self.__b[i] -= self.__learning_rate*delta_b[i]/x.shape[1]

        ## make sure self.__feedForward(x), self.__backPropagation(y) are not called in wrong context
        self.__a = None
        self.__z = None
        self.__delta = None        
        return success_count

    def reset(self):
        """
        reset weights, biases and dynamically changing class parameters
        """
        ## parameters for initialization
        random_multi_w = 0.01
        random_multi_b = 0.0001
        add = 1/2

        ## initialize weights 
        self.__w = [None]*(self.__hl+1)
        self.__w[0] = (random([self.__ahn, self.__ain])-add)*random_multi_w                 # w for second layer
        if self.__hl > 1:
            for j in range(1,self.__hl):                                                    # w for hidden layers
                self.__w[j] = (random([self.__ahn, self.__ahn])-add)*random_multi_w
        self.__w[len(self.__w)-1] = (random([self.__aon, self.__ahn])-add)*random_multi_w   # w for last layer

        ## initialize bias
        self.__b = [None]*(self.__hl+1)
        for j in range(0,self.__hl):
            self.__b[j] = (random(self.__ahn)-add)*random_multi_b
        self.__b[len(self.__b)-1] = (random(self.__aon)-add)*random_multi_b

        ## dynamically changing parameters
        ## next three parameters make sure self.__feedForward(x), self.__backPropagation(y) are not called in wrong context
        self.__a = None
        self.__z = None
        self.__delta = None 
        self.__learning_rate = None

    def train(self, x, y, max_steps=5000):
        """
        input:
          - x: numpy array with dimensions (amount_input_neurons, sample size), input for net
          - y: numpy array with dimensions (amount_output_neurons, sample size), desired output
        output: 
          - converged: boolen, True if success_percentage is reached, False else
          - steps: after how many iterations did the program stop 
        """
        ## note that numpy arrays are passed by reference, so passing x,y as often as I do it here is not inefficient
        assert(x.shape[0]==self.__ain and y.shape[0]==self.__aon and x.shape[1]==y.shape[1] and x.shape[1]!=0), "invalid input in train"
        converged = False
        self.__learning_rate = 0.5
        sp_tmp = 0
        sp_cnt = 0

        for steps in range(0, max_steps):
            patch = range(0, x.shape[1], min(self.mini_patch_size, x.shape[1]-1))
            mean_sc = 0
            ## split data into mini patches and fit w, b to input and output data of it
            for j in range(0,len(patch)-1):
                mean_sc += self._fit(x[:,patch[j]:patch[j+1]], y[:,patch[j]:patch[j+1]])
            print("percentage of correctly predicted numbers after {} steps:".format(steps+1),float(mean_sc)/x.shape[1]*100)
            mean_sc = float(mean_sc)/x.shape[1]

            ## change learning rate depending on success rate
            if mean_sc >= .90:
                self.__learning_rate = 0.1
            ## count how often success rate doesn't change
            if mean_sc == sp_tmp: sp_cnt+=1
            else: sp_cnt=0
            sp_tmp = mean_sc
            ## break if converged or no progress is made
            if mean_sc >= self.success_percentage:
                converged = True
                break
            elif sp_cnt >=70: # break if success_percentage is not changing
                converged = False
                break
            elif sp_cnt >=20: # slow down learning rate if nothing is changing
                self.__learning_rate = 0.1
        
        print("stopped after {} steps".format(steps+1))
        return [converged, steps]


    def test(self,x,y):
        """
        input: 
          - x: numpy array with dimensions (amount_input_neurons, sample size), input for net
          - y: numpy array with dimensions (amount_output_neurons, sample size), desired output
        output: percentage of correctly identified digits
        """
        assert(x.shape[0]==self.__ain and y.shape[0]==self.__aon and x.shape[1]==y.shape[1] and x.shape[1]!=0), "invalid input in test"
        self.__a = [None]*(self.__hl+2)
        self.__z = [None]*(self.__hl+1)
        success_count = 0
        for j in range(0, x.shape[1]):
            self.__feedForward(x[:,j])
            success_count+=int(np.argmax(y[:,j])==np.argmax(self.__a[len(self.__a)-1]))
        ret_val = success_count/x.shape[1]*100
        print("classified {} percent of test digits correctly".format(ret_val))
        return ret_val

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
        """
        input:
          - x:     numeric, numpy array or scalar
          - deriv: boolean, True if derivation of function is desired, False else
          - func:  string, "sigmoid" or "relu"
        output: numpy array, func(x) or func'(x) if deriv==True
        """
        x = np.array(x)
        if func=="sigmoid":
            if deriv==False:
                ret_val = 1/(1+np.exp(-x))
            else:
                ret_val = NeuronalNet.sigma(x, deriv=False, func="sigmoid")*(1-NeuronalNet.sigma(x, deriv=False, func="sigmoid"))
        elif func=="relu":
            if deriv==False:
                ret_val = x
                ret_val[ret_val<0]=0
            else:
                ret_val=np.zeros(len(x))
                ret_val[x>0]=1
        else:
            assert(False), "Invalide Function"
        return ret_val
