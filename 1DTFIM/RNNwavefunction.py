# SUccess to run
# 1D Ising model, 1D RNN model
'''
重大改變
* initialize用的 zero state 改法：rnn_state = self.rnn.get_initial_state(batch_size = self.numsamples, dtype = tf.float32)
* 原：self.rnn = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell(units[n]) for n in range(len(units))])
  後：self.rnn = tf.keras.layers.StackedRNNCells([cell(units[n]) for n in range(len(units))])
'''
#This is an implementation of the 1D pRNN wavefunction without a parity symmetry
# hua:2
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf
#hua
import numpy as np
import random

class RNNwavefunction(object):
    # units=units, cell=tf.compat.v1.nn.rnn_cell.GRUCell, seed = seed) #contains the graph with the RNNs
    # Hua: change_tf.nn.relu -> tf.nn.elu
    def __init__(self, systemsize, cell = tf.compat.v1.nn.rnn_cell.GRUCell, activation = tf.nn.elu, units=[10], scope='RNNwavefunction', seed = 111):
        """
        Hua: 
            systemsize:  N (spin number)
            cell:        tensorflow RNN cell
            units:       number of hidden units per RNN layer //note: this is a "list"
            scope:       the name of the name-space scope (str)
                         
        """
        #print('hua, parameters check:')
        #print('units = ', units )
        self.graph = tf.Graph()
        # Hua: A context manager for defining ops that creates variables (layers).
        self.scope = scope # Label of the RNN wavefunction
        self.N = systemsize 

        random.seed(seed)  # pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        # Defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope, reuse = tf.compat.v1.AUTO_REUSE):
                tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator
                # Define the RNN cell where units[n] corresponds to the number of memory units in each layer n
                # Hua: 因為hidden units是寫成一個list，因此要分別指定units的部分給不同的RNN cell 作為hidden units
                '''
                改：
                根據的網址：https://github.com/tensorflow/tensorflow/issues/28216
                那這個網址裡有提到tf.contrib.rnn.MultiRNNCell is an RNNCell, 
                while tf.keras.layers.RNN is an RNN API (like tf.nn.dynamic_rnn) instead of a kind of RNNCell. 
                An RNNCell defines how a single time step is computed, 
                while an RNN API loops over all time steps with a given RNNCell. 
                So you need tf.keras.layers.StackedRNNCells(which is a kind of RNNCell) instead of tf.keras.layers.RNN. 
                '''
                self.rnn = tf.keras.layers.StackedRNNCells([cell(units[n]) for n in range(len(units))])
                #self.rnn = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell(units[n]) for n in range(len(units))])
                # Define the Fully-Connected layer followed by a Softmax
                '''
                官網上的定義：Functional interface for the densely-connected layer.
                且：This layer implements the operation: outputs = activation(inputs * kernel + bias) 
                where activation is the activation function passed as the activation argument (if not None), 
                kernel is a weights matrix created by the layer, 
                and bias is a bias vector created by the layer (only if use_bias is True).
                補：name = String, the name of the layer.
                default中定義：reuse: none
                改：
                Hua: 或許可以改成這種，盡量不要call v1的話：y = tf.nn.softmax(tf.matmul(x, W) + b)
                '''
                #self.dense = tf.nn.softmax(tf.matmul(x, W) + b)
                self.dense = tf.compat.v1.layers.Dense( 2, activation = tf.nn.softmax, name = 'wf_dense') 

    # tf sample 7494 pytorch的forward，也就是網路的節夠是怎麼連接的 
    def sample (self, numsamples, inputdim):
        """
            將所有測資用 probability distribution parametrized by a recurrent network
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            parameters:
                numsamples:  總共有多少比測資      
                inputdim:    就是spin的dimension # hilbert space dimension of one spin 
            ------------------------------------------------------------------------
            returns:      
                samples:         tf.Tensor of shape (numsamples,systemsize)
                                 the samples in integer encoding
                                 就是預測出來的新的一組spin configuration
        """
        # 新開一張 graph 使用的意思 # with self.graph.as_default(): 
        with self.graph.as_default(): #Call the default graph, used if willing to create multiple graphs.
            samples = []
            # tf.compat.v1.variable_scope: A context manager for defining ops that creates variables (layers).
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                # b: 因為一開始的h[0]和sigma[0]都是偽造給予的，所以先亂給，給 0
                b = np.zeros((numsamples, inputdim)).astype(np.float64)
                # b = state of sigma_0 for all the samples
                # 其中，參數b其實就是最開始的input：sigma[0]
                # 並將input轉乘tf的形式
                inputs = tf.constant(dtype = tf.float32, value = b, shape = [numsamples, inputdim]) # Feed the table b in tf.

                self.inputdim = inputs.shape[1]
                self.outputdim = self.inputdim
                self.numsamples = (inputs.shape[0])

                # hua: debug1_okok: try:: get_initial_state(inputs=None, batch_size=None, dtype=None)
                rnn_state = self.rnn.get_initial_state(batch_size = self.numsamples, dtype = tf.float32) #Initialize the RNN hidden state
                #rnn_state = self.rnn.zero_state(self.numsamples, dtype = tf.float32) #Initialize the RNN hidden state
                #zero state returns a zero filled tensor withs shape = (self.numsamples, num_units)

                for n in range(self.N):
                    # self.rnn：丟進multiRNN裡面跑出多個hidden結果，其中hidden就用來預測每一層各自的sigma
                    # shape of output: [B, H]
                    # shape of new_state: (([B, H], [B, H]), ([B, H], [B, H]))
                    # keras stack的格式：output, new_state = self.rnn (x, state)
                    # rnn_output: hidden 的結果，其實就是h[i] for all the spins
                    rnn_output, rnn_state = self.rnn(inputs, rnn_state) # Compute the next hidden states
                    # 出來的output, h[i], 按照論文上，要再過一層softmax
                    output = self.dense(rnn_output) 
                    # 補：If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant. 
                    # In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
                    # 這部分轉換dimension的地方我沒有看懂，但我猜這裡可能會出現bug，在reshape的地方
                    sample_temp = tf.reshape(tf.random.categorical(logits = tf.math.log(output), num_samples=1), [-1,]) #Sample from the probability
                    # 我能確定的是：sample_temp其實就是預估出來的新spin的config，然後維度是[0.3, 0.7]
                    samples.append(sample_temp)
                    # 將結果經過one hot encoding後，作為下一次的input在餵進去
                    inputs = tf.one_hot(sample_temp, depth = self.outputdim) # outputdim = inputdim @line89
        
        # (self.N, num_samples) to (num_samples, self.N): 
        # Generate self.numsamples vectors of size self.N spin containing 0 or 1
        # 這行是在把新predict出來的spin弄成一維的，然後用tf的stack包起來丟出來ㄍ
        self.samples = tf.stack(values = samples, axis = 1) 
        # return : 預測出的spin config
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

                # hua, debug-2:
                rnn_state = self.rnn.get_initial_state(batch_size = self.numsamples, dtype = tf.float32)
                #rnn_state=self.rnn.zero_state(self.numsamples,dtype=tf.float32)

                for n in range(self.N):
                    rnn_output, rnn_state = self.rnn(inputs, rnn_state)
                    output=self.dense(rnn_output)
                    probs.append(output)
                    inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim),shape=[self.numsamples,self.inputdim])

            probs=tf.cast(tf.transpose(a=tf.stack(values=probs,axis=2),perm=[0,2,1]),tf.float64)
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(input_tensor=tf.math.log(tf.reduce_sum(input_tensor=tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

            return self.log_probs
