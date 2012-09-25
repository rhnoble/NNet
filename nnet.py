# A neural net with a single hidden layer
# Author: Harry Noble
# Modified: 11/22/11


import random
import math



class net:
    
    # Creates new neural net
    # Parameters: number of input nodes, number of hidden nodes, number of
    # output nodes

    def __init__ (self, ins, hids, outs):
        self.lastinput = []     # Stores most recent input to the net
        self.hids = []          # List of neurons in hidden layer
        for x in range(hids):
            self.hids.append(neuron(ins))
        self.outs = []          # List of neurons in output layer
        for x in range(outs):
            self.outs.append(neuron(hids))

    # Feeds input data through neural net
    # Parameter: Activation level of input nodes
    # Returns: Activation level of output nodes
            
    def go(self, inputs):
        self.lastinput = inputs
        hidsout = []
        for x in self.hids:
            hidsout.append(x.go(inputs))
        outs = []
        for x in self.outs:
            outs.append(x.go(hidsout))
        return outs

    # Backpropagates error to all output and hidden nodes
    # and updates weights accordingly
    # Parameter: Expected outcome of previous input
    
    def update(self, expected):
        # Finds error for output nodes
        for x in range(len(self.outs)):
            self.outs[x].error = self.outs[x].last * (1 - self.outs[x].last) *
            (expected[x] - self.outs[x].last)
        # Backpropagates error to hidden nodes
        for x in range(len(self.hids)):
            backerror = 0
            for y in range(len(self.outs)):
                backerror += (self.outs[y].error * self.outs[y].weights[x])
            self.hids[x].error = self.hids[x].last *
            (1-self.hids[x].last) * backerror
        # Updates weights
        for x in self.outs:
            for y in range(len(self.hids)):
                x.weights[y] += x.error * self.hids[y].last
        for x in self.hids:
            for y in range(len(self.lastinput)):
                x.weights[y] += x.error * self.lastinput[y]
    
class neuron:

    # Creates new neuron
    # Parameter: Number of nodes to recieve input from
    
    def __init__ (self, inputs):
        self.weights = []
        self.error = 0
        self.insum = 0
        self.last = 0
        for x in range(inputs):
            self.weights += [random.uniform(-6, 6)]

    # Sigmoid function to keep outputs between 0 and 1
    
    def sigmoid(self, x):
        return (1 / (1 + math.exp(-x)))

    # Feeds input data through neuron
    # Parameter: Activation level of input nodes
    # Returns: Activation level of neuron
            
    def go (self, inputs):
        self.insum = 0
        for x in range(len(inputs)):
            self.insum += (inputs[x] * self.weights[x])
        self.last = self.sigmoid(self.insum)
        return self.last


