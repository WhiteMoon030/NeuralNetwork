import numpy as np
import scipy.special 

class FeedForward3:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learningrate):
        '''Create a new instance of a three-layer neural network.'''
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        self.l_rate = learningrate

        # uses the sigmoid function as the activation function
        self.activation_function = lambda x: scipy.special.expit(x)


    def init_weights(self):
        '''Initialize the weights of the network using the normal distribution.'''
        self.w_ih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.w_ho = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

    def load_weights(self, w_ih_path, w_ho_path):
        self.w_ih = np.genfromtxt(w_ih_path, delimiter=",")
        self.w_ho = np.genfromtxt(w_ho_path, delimiter=",")

    def train(self, inputs_list, targets_list):
        '''Train the network based on a given input and the expected output.'''
        targets = np.array(targets_list, ndmin=2).T

        i_layer, h_layer, o_layer = self.query(inputs_list, all_layer=True)

        # calculate the error
        error_output = targets - o_layer
        error_hidden = np.transpose(self.w_ho) @ error_output

        # update the weights
        self.w_ho += self.l_rate * np.dot((error_output * o_layer * (1.0 - o_layer)), np.transpose(h_layer)) 
        self.w_ih += self.l_rate * np.dot((error_hidden * h_layer * (1.0 - h_layer)), np.transpose(i_layer)) 

        pass
    def query(self, inputs_list, all_layer = False):
        '''
        Queries an input vector through the network and returns the output layer.\n
        If all_layer is set to True, the outputs of all layers are returned, including the input layer.
        '''

        inputs_array = np.array(inputs_list, ndmin=2).T

        hidden_layer = self.activation_function(np.dot(self.w_ih, inputs_array))
        output_layer = self.activation_function(np.dot(self.w_ho, hidden_layer))
        
        if all_layer:
            return inputs_array, hidden_layer, output_layer
        return output_layer
