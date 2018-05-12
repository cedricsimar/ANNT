# -*- coding: utf-8 -*-

from collections import deque

from tf_decorator import *
from settings import Settings
from exceptions import InvalidNumberOfEdges, ImpossibleToBuild

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

        # Tensorflow graph
        # self.graph = tf.Graph()

        # DNA of the Neural Network
        self.dna = dna

        # Graph vertex root
        self.root = self.dna.vertices[self.dna.input_vertex_id]

        # tensors created by edges and vertices from the DNA
        self.vertices_tensor = {}
        self.edges_tensor = {}

        # network building queue
        self.queue = deque([])

        # retrieve input dimensionality
        self.input_dim = len(self.dna.input_shape)

        # creating input placeholder
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
        self.prediction_error
        self.optimize
        self.loss
            

        # # Initialize input placeholder to assign values to weights and biases
        # with tf.variable_scope("input_assignment"):

        #     self.l_param_input = {}
        #     self.assign_operator = {}
        #     for variable_name in self.learning_parameters.keys():
        #         self.l_param_input[variable_name] = tf.placeholder(
        #             tf.float32,
        #             self.learning_parameters[variable_name].get_shape().as_list(),
        #             name=variable_name)

        #         try:  # If mutable tensor (Variable)
        #             self.assign_operator[variable_name] = self.learning_parameters[variable_name].assign(
        #                 self.l_param_input[variable_name])
        #         except AttributeError as e:
        #             print(e)

    

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
        self.vertices_tensor[self.dna.input_vertex_id] = input_layer


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
                    # if the queue is empty raise exception to avoid infinite loop
                    if(len(self.queue) > 0):
                        self.queue.append(graph_object)
                    else:
                        raise ImpossibleToBuild()

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
                    # if the queue is empty raise exception to avoid infinite loop
                    if(len(self.queue) > 0):
                        self.queue.append(graph_object)
                    else:
                        raise ImpossibleToBuild()
        
        return(tensor)


    def from_vertex_to_tensor(self, v):

        # It is assumed that inputs dimension of all actions have been properly 
        # checked during the mutation phase

        # First we check if all input edge tensors have already been created
        for input_edge in v.edges_in:
            if input_edge.id not in self.edges_tensor:
                return(None)

        # Check if the action type matches with the number of input edges
        if (v.action == Settings.NO_ACTION and len(v.edges_in) > 1) or (v.action != Settings.NO_ACTION and len(v.edges_in) < 2):
            raise InvalidNumberOfEdges()
        
        # sequentially check the vertex attributes and create the tensors accordingly
        # action -> batch normalization -> activation -> max-pooling -> dropout

        # No action, sum or concatenation
        if v.action == Settings.NO_ACTION:

            # ouput of "No action" if the input tensor
            tensor = self.edges_tensor[v.edges_in[0].id]

        else:
            
            # compute the list of input tensors
            input_tensors = [self.edges_tensor[e.id] for e in v.edges_in]

            # apply the action
            if v.action == Settings.SUM:

                tensor = input_tensors[0]
                for i in range(1, len(input_tensors)):
                    tensor = tf.add(tensor, input_tensors[i])
            
            elif v.action == Settings.CONCATENATION:

                tensor = tf.concat(input_tensors, axis = 1)


        # batch normalization
        # TODO or not TODO

        # activation / non-linearity
        if v.activation == Settings.RELU:

            tensor = tf.nn.relu(tensor)

        # max pooling
        if v.max_pooling == Settings.USE_MAX_POOLING:
            tensor = tf.layers.max_pooling2d(inputs=tensor, pool_size=Settings.DEFAULT_POOLING_SHAPE, strides=Settings.DEFAULT_POOLING_STRIDE)
        
        # flatten
        if v.flatten == Settings.FLATTEN:
            tensor = tf.layers.flatten(tensor)

        # dropout
        if v.dropout == Settings.USE_DROPOUT:
            tensor = tf.layers.dropout(inputs=tensor, rate=Settings.DROPOUT_RATE)

        return (tensor)
    

    def from_edge_to_tensor(self, e):

        # First we check if the vertex tensor has already been created
        if e.from_vertex.id not in self.vertices_tensor:
            return (None)

        tensor = self.vertices_tensor[e.from_vertex.id]
        
        if e.type == Settings.FULLY_CONNECTED:

            tensor = tf.layers.dense(tensor, e.units, use_bias=True)

        elif e.type == Settings.CONVOLUTIONAL:

            tensor=tf.layers.conv2d(tensor, e.kernels, e.kernel_shape, e.stride, padding="same",
                                    kernel_initializer = glorot_uniform_initializer())

        return (tensor)
        

    # @define_scope
    # def predict_proba(self):
    #     return(tf.sigmoid(self.predict))
    
    @define_scope
    def prediction_error(self):

        """ Return the mean prediction error """

        prediction_errors = tf.not_equal(tf.argmax(self.predict, 1), tf.argmax(self.labels, 1))
        return (tf.reduce_mean(tf.cast(prediction_errors, tf.float32)))


    @define_scope
    def loss(self):

        """ Return the mean loss """

        self.batch_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.predict)
        self.mean_loss = tf.reduce_mean(self.batch_loss, name="mean_loss")
        return (self.mean_loss)
        

    @define_scope
    def optimize(self):

        self.optimizer = tf.train.MomentumOptimizer(
            learning_rate = Settings.LEARNING_RATE,
            momentum = Settings.MOMENTUM,
            use_nesterov = True
        )
        return(self.optimizer.minimize(self.loss))


    def get_graph(self):
        return(self.graph)
    

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
