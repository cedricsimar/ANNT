# -*- coding: utf-8 -*-

from collections import deque

from tf_decorator import *
from settings import Settings

import tensorflow as tf
from tensorflow.python.ops import array_ops as tf_array_ops
from tensorflow.python.ops.init_ops import glorot_uniform_initializer

from vertex import Vertex
from edge import Edge

class NN:

    def __init__(self, dna):

        """
        Neural Network build using a DNA object
        """

        # DNA of the Neural Network
        self.dna = dna

        # Graph vertex root
        self.root = self.dna.vertices[0]

        # tensors created by edges and vertices from the DNA
        self.vertices_tensor = {}
        self.edges_tensor = {}

        # network building queue
        self.queue = deque([])

        # retrieve input dimensionality
        self.input_dim = len(self.dna.input_shape)

        # receiving input placeholder
        if self.input_dim == 1:
            # 1D input
            self.input = tf.placeholder(tf.float32, [None, self.dna.input_shape[0]])
        elif self.input_dim == 2:
            # 2D input
            self.input = tf.placeholder(tf.float32, [None, self.dna.input_shape[0], self.dna.input_shape[1]])

        self.labels = tf.placeholder(tf.float32, [None, self.dna.output_shape])

        # Network settings
        self.norm_eps = Settings.NORMALIZATION_EPSILON

        # weights and biases dictionary
        self.learning_parameters = {}
        self.layers = {}

        # initialize tensorflow graph
        self.predict
        self.optimize
        self.loss

        # Initialize input placeholder to assign values to weights and biases
        with tf.variable_scope("input_assignment"):

            self.l_param_input = {}
            self.assign_operator = {}
            for variable_name in self.learning_parameters.keys():
                self.l_param_input[variable_name] = tf.placeholder(
                    tf.float32,
                    self.learning_parameters[variable_name].get_shape().as_list(),
                    name=variable_name)

                try:  # If mutable tensor (Variable)
                    self.assign_operator[variable_name] = self.learning_parameters[variable_name].assign(
                        self.l_param_input[variable_name])
                except AttributeError as e:
                    print(e)

    def from_vertex_to_tensor(self, v):

        # It is assumed that inputs dimension of all actions have been properly 
        # checked during the mutation phase

        v = Vertex()

        """ D'abord vérifier qu'on a tous les tensors d'input disponibles, sinon return(None) """

        # action -> batch normalization -> activation -> max-pooling -> dropout

        # No action, sum or concatenation

        """ Check séquentiel des attributs du Vertex """

        return (tensor)
    
    def from_edge_to_tensor(self, e):

        return (tensor)
        

    @define_scope
    def predict(self):

        """
        First handle the input vertex (root) to the neural network separately
        because it doesn't have any incoming edges
        """
        # reshape input to 4d tensor [batch, shape_x, shape_y, 1]
        if self.input_dim == 1:
            # 1D input
            input_layer = tf.reshape(self.input, [-1, self.dna.input_shape[0], 1])
        elif self.input_dim == 2:
            # 2D input
            input_layer = tf.reshape(self.input, [-1, self.dna.input_shape[0], self.dna.input_shape[1], 1])        

        # push all outgoing edges to the queue
        for edge_out in self.root.edges_out:
            self.queue.append(edge_out)

        # add the resulting tensor in a dictionary using the vertex id
        self.vertices_tensor[0] = input_layer


        """
        Iteratively build the Neural Network layers following the DNA graph
        """
        while(len(self.queue) > 0):
            
            graph_object = self.queue.popleft()

            if(graph_object.is_vertex()):

                # create tensor from vertex object
                tensor = self.from_vertex_to_tensor(graph_object)

                # save tensor if it was successfully created
                if tensor is not None:
                    
                    # add the resulting tensor in the vertices dictionary
                    self.vertices_tensor[graph_object.id] = tensor

                    # push all outgoing edges to the queue
                    for edge_out in graph_object.edges_out:
                        self.queue.append(edge_out)
                
                else:
                    # put vertex back in queue
                    # more tensors have to be created before this one 
                    self.queue.append(graph_object)

            else:

                # create tensor from edge object
                tensor = self.from_edge_to_tensor(graph_object)

                # save tensor if it was successfully created
                if tensor is not None:

                    # add the resulting tensor in the edges dictionary
                    self.edges_tensor[graph_object.id] = tensor

                    # push the destination vertex to the queue
                    self.queue.append(graph_object.to_vertex)

                else:
                    # put edge back in queue
                    # more tensors have to be created before this one 
                    self.queue.append(graph_object)


        """
        Layer 1: 1D spatial convolution over the channels to condense NUM_CHANNELS to
                 the number of the convolutional filters (thus reducing the input dimenstion)
                 Linear transformation, no activation function used
        """
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=4, kernel_size=[Settings.NUM_CHANNELS, 1],
                                 strides = (1, 1), padding="valid", kernel_initializer=glorot_uniform_initializer(),
                                 activation=None)

        print("conv1 shape: ", conv1.get_shape())
                                 
        """
        Layer 2: 1D temporal convolution over the raw signal of each channel
                 Activation function: ReLu
                 Max pooling 
        """
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[1, 9], strides=(1, 1),
                                 padding = "same", kernel_initializer = glorot_uniform_initializer(),
                                 activation = tf.nn.leaky_relu)

        print("conv2 shape: ", conv2.get_shape())

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(1, 4), strides=(1, 4))
        # no dropout in the first layer

        print("pool2 shape: ", pool2.get_shape())

        """
        Layer 3: Second 1D temporal convolution
                 Activation function: ReLu
                 Max pooling
                 Flatten the output of the max pooling layer to get the input of the FC layer
                 Dropout layer with p = 0.5
        """
        conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[1, 9], strides=(1, 1),
                                 padding = "same", kernel_initializer = glorot_uniform_initializer(),
                                 activation = tf.nn.leaky_relu)
        
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(1, 4), strides=(1, 4))

        flat3 = tf.layers.flatten(pool3)
        
        drop3 = tf.layers.dropout(inputs=flat3, rate=Settings.DROPOUT_RATE)
        

        print("drop3 shape: ", drop3.get_shape())

        """
        Layers 4 and 5: Standard Fully Connected layers separated by Dropout layers
        """
        fc4 = tf.layers.dense(inputs=drop3, units=1024, activation=tf.nn.relu, use_bias=True,
                              kernel_initializer = glorot_uniform_initializer())
        drop4 = tf.layers.dropout(inputs=fc4, rate=Settings.DROPOUT_RATE)

        fc5 = tf.layers.dense(inputs=drop4, units=1024, activation=tf.nn.relu, use_bias=True,
                              kernel_initializer = glorot_uniform_initializer())
        drop5 = tf.layers.dropout(inputs=fc5, rate=Settings.DROPOUT_RATE)

        
        """
        Layer 6: Output layer
        """
        logits = tf.layers.dense(inputs=drop5, units=Settings.NUM_EVENTS,
                               kernel_initializer = glorot_uniform_initializer(),
                               activation = None)

        # using sigmoid cross entropy (not mutually exclusive) with logits so no need of an
        # activation function at the end of the CNN
        
        return(logits)


    @define_scope
    def predict_proba(self):
        return(tf.sigmoid(self.predict))
    
    @define_scope
    def accuracy(self):
        self.predictions = tf.round(self.predict_proba)
        self.correct_predictions = tf.cast(tf.equal(self.predictions, self.labels), tf.float32)
        return(tf.reduce_mean(self.correct_predictions))

    @define_scope
    def optimize(self):

        self.optimizer = tf.train.MomentumOptimizer(
            learning_rate = Settings.LEARNING_RATE,
            momentum = Settings.MOMENTUM,
            use_nesterov = True
        )
        return(self.optimizer.minimize(self.loss))


    @define_scope
    def loss(self):
        """ Return the mean error """

        self.error = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.predict)
        self.mean_error = tf.reduce_mean(self.error, name="mean_error")
        return(self.mean_error)

    
    def get_value(self, var_name, tf_session):
        """
        Return the value of the tf variable named [var_name] if it exists, None otherwise
        """

        if var_name in self.learning_parameters:

            value = tf_session.run(self.learning_parameters[var_name])

        elif var_name in self.layers:

            value = tf_session.run(self.layers[var_name])

        else:
            print("Unknown DQN variable: " + var_name)
            assert(0)  # <3

        return(value)

    def set_value(self, var_name, new_value, tf_session):
        """
        Set the value of the tf variable [var_name] to [new_value]
        """

        if(var_name in self.assign_operator):

            tf_session.run(
                self.assign_operator[var_name], {
                    self.l_param_input[var_name]: new_value})
        else:
            print("Thou shall only assign learning parameters!")
