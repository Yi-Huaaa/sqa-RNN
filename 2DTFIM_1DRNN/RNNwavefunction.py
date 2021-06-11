# Date: 20210611 17:45, hua
#2D Ising model, 1D RNN model
# Success to run

import tensorflow as tf
import numpy as np
import random


class RNNwavefunction(object):
    def __init__(self,systemsize_x, systemsize_y, cell = tf.compat.v1.nn.rnn_cell.GRUCell,activation=tf.nn.elu,units=[10],scope='RNNwavefunction',seed = 111):
        """
            systemsize_x:  int
                         number of sites for x-axis
            systemsize_y:  int
                         number of sites for y-axis         
            cell:        a tensorflow RNN cell
            activation:  activation function used for the RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            seed:        pseudo-random number generator 
        """
        
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.Nx=systemsize_x #x_size of the lattice
        self.Ny=systemsize_y #y_size of the lattice

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                # hua 1, 改：
                tf.random.set_seed(seed)
                #tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator
                #Define the RNN cell where units[n] corresponds to the number of memory units in each layer n

                # hua 1, 改：
                self.rnn = tf.keras.layers.StackedRNNCells([cell(units[n]) for n in range(len(units))])                
                #self.rnn=tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell(units[n]) for n in range(len(units))])
                
                # hua 2, 改：
                self.dense = tf.compat.v1.layers.Dense( 2, activation=tf.nn.softmax, name='wf_dense') #Fully-Connected, Softmax
                #self.dense = tf.compat.v1.layers.Dense(2, activation=tf.nn.softmax, name='wf_dense', dtype = tf.float64) #Fully-Connected, Softmax

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

            samples:         tf.Tensor of shape (numsamples,systemsize_x*systemsize_y)
                             the samples in integer encoding
        """
        with self.graph.as_default(): #Call the default graph, used if not willing to create multiple graphs.
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                b=np.zeros((numsamples,inputdim)).astype(np.float64)
                #b = state of sigma_0 for all the samples
                
                #hua 2, 改：
                inputs=tf.constant(dtype=tf.float32,value=b,shape=[numsamples,inputdim]) #Feed the table b in tf.
                #inputs=tf.constant(dtype=tf.float64,value=b,shape=[numsamples,inputdim]) #Feed the table b in tf.
                #Initial input to feed to the rnn

                self.inputdim=inputs.shape[1]
                self.outputdim=self.inputdim
                self.numsamples=inputs.shape[0]

                samples=[]

                #hua 1, 改：
                # hua 2, 改：
                rnn_state = self.rnn.get_initial_state(batch_size = self.numsamples, dtype = tf.float32) 
                #rnn_state = self.rnn.get_initial_state(batch_size = self.numsamples, dtype = tf.float64) 

                
                for ny in range(self.Ny): #Loop over the lattice in a snake shape
                    for nx in range(self.Nx): 
                        rnn_output, rnn_state = self.rnn(inputs, rnn_state)
                        output=self.dense(rnn_output)
                        sample_temp=tf.reshape(tf.random.categorical(logits=tf.math.log(output),num_samples=1),[-1,])
                        samples.append(sample_temp)
                        # hua 2, 改：
                        inputs=tf.one_hot(sample_temp,depth=self.outputdim)
                        #inputs=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)

        self.samples=tf.stack(values=samples,axis=1) # (self.N, num_samples) to (num_samples, self.N): Generate self.numsamples vectors of size self.N spin containing 0 or 1

        return self.samples

    def log_probability(self,samples,inputdim):
        """
            calculate the log-probabilities of ```samples``
            ------------------------------------------------------------------------
            Parameters:
            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize_x*system_size_y)
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
            # hua 2, 改：
            a=tf.zeros(self.numsamples, dtype=tf.float32)
            b=tf.zeros(self.numsamples, dtype=tf.float32)
            #a=tf.zeros(self.numsamples, dtype=tf.float64)
            #b=tf.zeros(self.numsamples, dtype=tf.float64)

            inputs=tf.stack([a,b], axis = 1)

            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                probs=[]

                #hua 1, 改：
                # hua 2, 改：
                rnn_state = self.rnn.get_initial_state(batch_size = self.numsamples, dtype = tf.float32) 
                #rnn_state = self.rnn.get_initial_state(batch_size = self.numsamples, dtype = tf.float64) 

                for ny in range(self.Ny):
                    for nx in range(self.Nx):
                        rnn_output, rnn_state = self.rnn(inputs, rnn_state)
                        output=self.dense(rnn_output)
                        probs.append(output)
                        # hua 2, 改： 
                        inputs = tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(ny*self.Nx+nx)],size=[np.int32(-1),np.int32(1)]), shape=[self.numsamples]),depth=self.outputdim),shape=[self.numsamples,self.inputdim])
                        #inputs = tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(ny*self.Nx+nx)],size=[np.int32(-1),np.int32(1)]), shape=[self.numsamples]),depth=self.outputdim,dtype = tf.float64),shape=[self.numsamples,self.inputdim])

            # hua 2, 改：
            probs=tf.cast(tf.transpose(a=tf.stack(values=probs,axis=2),perm=[0,2,1]),tf.float64)
            #probs=tf.transpose(a=tf.stack(values=probs,axis=2),perm=[0,2,1])
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(input_tensor=tf.math.log(tf.reduce_sum(input_tensor=tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

            return self.log_probs
