# Date: 20210611 17:46, hua
# Success to run
#2D Ising model, 2D RNN model

import tensorflow as tf
import numpy as np
import random
# hua 3, 改: 
# 一定要加這條否則會一堆垃圾error
tf.compat.v1.disable_eager_execution()

class RNNwavefunction(object):
    def __init__(self,systemsize_x, systemsize_y,cell=None,units=[10],scope='RNNwavefunction',seed = 111):
        """
            systemsize_x:  int
                         number of sites for x-axis
            systemsize_y:  int
                         number of sites for y-axis         
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            seed:        pseudo-random number generator 
        """
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.Nx=systemsize_x #size of x direction in the 2d model
        self.Ny=systemsize_y

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                # hua
                tf.random.set_seed(seed)
                #tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator
                #Define the RNN cell where units[n] corresponds to the number of memory units in each layer n
                
                # 我猜這裡應該要改成float32, 才可以應用tendorflow2.x verison's precision
                self.rnn = cell(num_units = units[0], num_in = 2 ,name="rnn_"+str(0),dtype=tf.float32)
                #self.rnn = cell(num_units = units[0], num_in = 2 ,name="rnn_"+str(0),dtype=tf.float64)
                self.dense = tf.compat.v1.layers.Dense(2, activation=tf.nn.softmax, name='wf_dense')
                # self.dense = tf.compat.v1.layers.Dense(2, activation=tf.nn.softmax, name='wf_dense', dtype = tf.float64)

    def sample(self,numsamples,inputdim):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:

            numsamples:      int
                             
                             
                             
                             samples to be produced
            inputdim:        int
                             hilbert space dimension of one spin

            ------------------------------------------------------------------------
            Returns:      

            samples:         tf.Tensor of shape (numsamples,systemsize_x, systemsize_y)
                             the samples in integer encoding
        """

        with self.graph.as_default(): #Call the default graph, used if not willing to create multiple graphs.
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

                #Initial input to feed to the 2drnn

                self.inputdim = inputdim
                self.outputdim = self.inputdim
                self.numsamples = numsamples


                samples=[[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
                rnn_states = {}
                inputs = {}
                # hua 修改 input dimension:
                for ny in range(self.Ny): #Loop over the boundary
                    if ny%2==0:
                        nx = -1
                        # print(nx,ny)
                        
                        # hua 2，改：
                        rnn_states[str(nx)+str(ny)] = self.rnn.get_initial_state(batch_size = self.numsamples, dtype = tf.float32) 
                        #rnn_states[str(nx)+str(ny)] = self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                        inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) 

                    if ny%2==1:
                        nx = self.Nx
                        # print(nx,ny)
                        # hua 2，改：
                        rnn_states[str(nx)+str(ny)] = self.rnn.get_initial_state(batch_size = self.numsamples, dtype = tf.float32)                         
                        #rnn_states[str(nx)+str(ny)] = self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                        inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) 


                for nx in range(self.Nx): #Loop over the boundary
                    ny = -1
                    # hua 2，改：
                    rnn_states[str(nx)+str(ny)] = self.rnn.get_initial_state(batch_size = self.numsamples, dtype = tf.float32)                         
                    #rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) 

                '''
                for ny in range(self.Ny): #Loop over the boundary
                    if ny%2==0:
                        nx = -1
                        # print(nx,ny)
                        rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float64)
                        inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) 

                    if ny%2==1:
                        nx = self.Nx
                        # print(nx,ny)
                        rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float64)
                        inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) 


                for nx in range(self.Nx): #Loop over the boundary
                    ny = -1
                    rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float64)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) 
                '''
                #Begin sampling
                for ny in range(self.Ny): 

                    if ny%2 == 0:

                        for nx in range(self.Nx): #left to right

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)]), (rnn_states[str(nx-1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))

                            output=self.dense(rnn_output)
                            sample_temp=tf.reshape(tf.random.categorical(logits=tf.math.log(output),num_samples=1),[-1,])
                            samples[nx][ny] = sample_temp
                            # hua 1，改：
                            inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim)
                            #inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)


                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)]), (rnn_states[str(nx+1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))

                            output=self.dense(rnn_output)
                            sample_temp=tf.reshape(tf.random.categorical(logits=tf.math.log(output),num_samples=1),[-1,])
                            samples[nx][ny] = sample_temp
                            # hua 1，改：
                            inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim)
                            #inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)
        ## hua bug is here
        self.samples=tf.transpose(a=tf.stack(values=samples,axis=0), perm = [2,0,1])

        return self.samples

    def log_probability(self,samples,inputdim):
        """
            calculate the log-probabilities of ```samples``
            ------------------------------------------------------------------------
            Parameters:

            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize_x,system_size_y)
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
            self.outputdim=self.inputdim


            samples_=tf.transpose(a=samples, perm = [1,2,0])
            rnn_states = {}
            inputs = {}
            # hua, 改：
            for ny in range(self.Ny): #Loop over the boundary
                if ny%2==0:
                    nx = -1
                    
                    # hua 2，改：
                    rnn_states[str(nx)+str(ny)] = self.rnn.get_initial_state(batch_size = self.numsamples, dtype = tf.float32) 
                    #rnn_states[str(nx)+str(ny)] = self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) 

                if ny%2==1:
                    nx = self.Nx
                    # hua 2，改：
                    rnn_states[str(nx)+str(ny)] = self.rnn.get_initial_state(batch_size = self.numsamples, dtype = tf.float32) 
                    #rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) 


            for nx in range(self.Nx): #Loop over the boundary
                ny = -1
                # hua 2，改：
                rnn_states[str(nx)+str(ny)] = self.rnn.get_initial_state(batch_size = self.numsamples, dtype = tf.float32) 
                #rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) 

            '''
            for ny in range(self.Ny): #Loop over the boundary
                if ny%2==0:
                    nx = -1
                    rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float64)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) 

                if ny%2==1:
                    nx = self.Nx
                    rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float64)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) 


            for nx in range(self.Nx): #Loop over the boundary
                ny = -1
                rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float64)
                inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) 
            '''

            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                probs = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]

                #Begin estimation of log probs
                for ny in range(self.Ny):

                    if ny%2 == 0:

                        for nx in range(self.Nx): #left to right

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)]), (rnn_states[str(nx-1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))

                            output=self.dense(rnn_output)
                            sample_temp=tf.reshape(tf.random.categorical(logits=tf.math.log(output),num_samples=1),[-1,])
                            probs[nx][ny] = output
                            # hua 1，改：                            
                            inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim)
                            #inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim,dtype = tf.float64)

                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)]), (rnn_states[str(nx+1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))

                            output=self.dense(rnn_output)
                            sample_temp=tf.reshape(tf.random.categorical(logits=tf.math.log(output),num_samples=1),[-1,])
                            probs[nx][ny] = output
                            # hua 1，改：
                            inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim)
                            #inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim,dtype = tf.float64)
            # hua 2, 改：
            probs = tf.cast(tf.transpose(a=tf.stack(values=probs,axis=0),perm=[2,0,1,3]),tf.float64)
            #probs=tf.transpose(a=tf.stack(values=probs,axis=0),perm=[2,0,1,3])
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=tf.math.log(tf.reduce_sum(input_tensor=tf.multiply(probs,one_hot_samples),axis=3)),axis=2),axis=1)

            return self.log_probs
