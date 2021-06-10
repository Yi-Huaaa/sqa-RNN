#This is an implementation of the 1D pRNN wavefunction without a parity symmetry

import tensorflow as tf
import numpy as np
import random

class RNNwavefunction(object):
    def __init__(self,systemsize,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell,units=[10],scope='RNNwavefunction', seed = 111):
        """
            systemsize:  int
                         number of sites      
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            seed:        pseudo-random number generator 
        """
        
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.N=systemsize #Number of sites of the 1D chain

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator
                #Define the RNN cell where units[n] corresponds to the number of memory units in each layer n
                self.rnn=tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell(units[n]) for n in range(len(units))])
                self.dense = tf.compat.v1.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense') #Define the Fully-Connected layer followed by a Softmax

    def sample(self,numsamples,inputdim):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:
            numsamples:      int
                             number of samples to be produced
            inputdim:        int
                             hilbert space dimension of one spin 
            ------------------------------------------------------------------------
            Returns:      
            samples:         tf.Tensor of shape (numsamples,systemsize)
                             the samples in integer encoding
        """
        with self.graph.as_default(): #Call the default graph, used if not willing to create multiple graphs.
            samples = []
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                b=np.zeros((numsamples,inputdim)).astype(np.float64)
                #b = state of sigma_0 for all the samples

                inputs=tf.constant(dtype=tf.float32,value=b,shape=[numsamples,inputdim]) #Feed the table b in tf.
                #Initial input to feed to the rnn

                self.inputdim=inputs.shape[1]
                self.outputdim=self.inputdim
                self.numsamples=inputs.shape[0]

                rnn_state=self.rnn.zero_state(self.numsamples,dtype=tf.float32) #Initialize the RNN hidden state
                #zero state returns a zero filled tensor withs shape = (self.numsamples, num_units)

                for n in range(self.N):
                    rnn_output, rnn_state = self.rnn(inputs, rnn_state) #Compute the next hidden states
                    output=self.dense(rnn_output) #Apply the Softmax layer
                    sample_temp=tf.reshape(tf.random.categorical(logits=tf.math.log(output),num_samples=1),[-1,]) #Sample from the probability
                    samples.append(sample_temp)
                    inputs=tf.one_hot(sample_temp,depth=self.outputdim)

        self.samples=tf.stack(values=samples,axis=1) # (self.N, num_samples) to (num_samples, self.N): Generate self.numsamples vectors of size self.N spin containing 0 or 1

        return self.samples

    def log_probability(self,samples,inputdim):
        """
            calculate the log-probabilities of ```samples``
            ------------------------------------------------------------------------
            Parameters:
            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize)
                             containing the input samples in integer encoding
            inputdim:        int
                             dimension of the input space
            ------------------------------------------------------------------------
            Returns:
            log-probs        tf.Tensor of shape (number of samples,)
                             the log-probability of each sample
            """
        with self.graph.as_default():

            self.inputdim=inputdim
            self.outputdim=self.inputdim

            self.numsamples=tf.shape(input=samples)[0]
            a=tf.zeros(self.numsamples, dtype=tf.float32)
            b=tf.zeros(self.numsamples, dtype=tf.float32)

            inputs=tf.stack([a,b], axis = 1)

            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                probs=[]

                rnn_state=self.rnn.zero_state(self.numsamples,dtype=tf.float32)

                for n in range(self.N):
                    rnn_output, rnn_state = self.rnn(inputs, rnn_state)
                    output=self.dense(rnn_output)
                    probs.append(output)
                    inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim),shape=[self.numsamples,self.inputdim])

            probs=tf.cast(tf.transpose(a=tf.stack(values=probs,axis=2),perm=[0,2,1]),tf.float64)
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(input_tensor=tf.math.log(tf.reduce_sum(input_tensor=tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

            return self.log_probs
