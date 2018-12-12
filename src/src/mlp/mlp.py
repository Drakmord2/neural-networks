#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .neuron import Neuron
from .synapse import Synapse
import random

 
class MLP(object):
    """
    Multi-Layer Perceptron
    
    Parameters: alpha=Learning rate; numIn=Neurons on input layer;
                layers=Neurons of each hidden layer; numOut=Neurons on output layer;
                funcs=Tuple with activation function of each layer;
                weights=List with all synaptic weights;
    """
    
    def __init__(self, alpha, num_in, layers, num_out, funcs, weights, loss_func):
        """
        Build a network
        """
        self.type = "MLP"
        self.alpha = alpha
        self.inputs = []
        self.hidden = []
        self.outputs = []
        self.errors = []
        self.loss_func = loss_func
        self.loss = -1.0
        
        self.sanitize(num_in, layers, num_out, funcs, weights)
        self.init_neurons(num_in, layers, num_out, funcs, weights)

    def compute(self, inputs):
        """
        Given a set of inputs, Classify or Predict
        """
        self.get_inputs(inputs)
        self.feedforward()
        
        result = []
        for out in self.outputs:
            result.append(out.value)
        
        return result

    def train(self, example, outputs):
        """
        Network training
        """
        if (len(self.inputs)-1 != len(example)-outputs):
            raise Exception("Invalid input data format")
            
        self.get_inputs(example)
            
        expected = example.iloc[len(example)-outputs:]
        
        error = self.feedforward(expected)
        self.feedback(error)
        self.learning()
        self.errors += error
        
    def feedforward(self, expected=None):
        """
        feedForward algorithm
        
        Parameters: expected=expected output values
        """
        
        for layer in range(len(self.hidden)):
            for neuron in self.hidden[layer][1:]:    
                net = 0
                for syn in neuron.synapses:
                    if syn.end == neuron:
                        net += syn.start.value * syn.weight
                neuron.net = net
                neuron.evaluate()
                
        if expected is not None:
            expected = list(expected)
        
        error = []
        n = 0
        for neuron in self.outputs:
            net = 0
            for syn in neuron.synapses:
                if syn.end == neuron:
                    net += syn.start.value * syn.weight
            neuron.net = net
            neuron.evaluate()
            
            if expected is not None:
                error.append(expected[n] - neuron.value)
            n += 1
        
        return error
    
    def feedback(self, error):
        """
        Backpropagation algorithm
        
        Parameters: error=expect output minus derived output
        """
        n = 0
        for neuron in self.outputs:
            neuron.backpropagate(error[n])
            n += 1
        
        for layer in range(len(self.hidden)-1, -1, -1):
            for neuron in self.hidden[layer][1:]: 
                neuron.backpropagate()
    
    def learning(self):
        """
        Learning algorithm
        
        W(new) = W(old) + alpha * sensitivity * f(net)
        """
        for layer in range(len(self.hidden)):
            for neuron in self.hidden[layer]:
                for syn in neuron.synapses:
                    if syn.start == neuron:
                        syn.weight = syn.weight + (self.alpha * syn.end.sensitivity * neuron.value) 
                        
        for neuron in self.inputs:
            for syn in neuron.synapses:
                if syn.start == neuron:
                    syn.weight = syn.weight + (self.alpha * syn.end.sensitivity * neuron.value)        

    def get_inputs(self, data):
        for i in range(1, len(self.inputs)):
            neuron = self.inputs[i]
            neuron.value = data.iloc[i-1]

    def get_loss(self):
        if self.loss_func == 'MSE':
            return self.get_mse()
        
        if self.loss_func == 'CE':
            return self.get_ce()
        
    def get_mse(self):
        """
        Mean Square Error
        
        MSE = 1/N * ∑ e^2
        """
        mse = 0
        for error in self.errors:
            mse += error ** 2
        mse = mse / len(self.errors)
        
        self.loss = mse
        
        return mse
    
    def get_ce(self):
        """
        Cross Entropy
        
        CE = 
        """
        ce = 0
        for error in self.errors:
            ce += 0
        
        self.loss = ce
        
        return ce

    def get_optimal_weights(self):
        weights = []
        
        for neuron in self.inputs:
            for syn in neuron.synapses:
                if syn.start == neuron:
                    weights.append(syn.weight)
                    
        for layer in self.hidden:
            for neuron in layer:
                for syn in neuron.synapses:
                    if syn.start == neuron:
                        weights.append(syn.weight)
        
        return weights

    def get_output(self):
        outputs = []
        for out in self.outputs:
            outputs.append(out.value)
        
        return outputs

    def init_neurons(self, num_in, layers, num_out, funcs, weights):
        # Threshold neurons
        neuron = Neuron('0', '1', 1.0) #Input
        self.inputs.append(neuron)
        for n in range(len(layers)): #Hidden layers
            neuron = Neuron('0', str(n+2), 1.0)
            self.hidden.append([neuron])
        
        # Neurons
        for n in range(num_in): #Input
            idn = str(n+1)
            neuron = Neuron(idn, '1')
            
            self.inputs.append(neuron)
        
        for l in range(len(layers)): #Hidden layers
            neurons = []
            for n in range(layers[l]):
                idn = str(n+1)
                neuron = Neuron(idn, str(l+2), func=funcs[l])
                neurons.append(neuron)
                
            self.hidden[l] += neurons

        for n in range(num_out): #Output
            idn = str(n+1)
            neuron = Neuron(idn, str(len(layers)+2), func=funcs[len(layers)])
            
            self.outputs.append(neuron)
            
        # Connections
        if weights:
            self.init_synapses(weights)
        else:
            self.init_synapses(self.initial_weights())

    def init_synapses(self, weights):
        index = 0
        num_in = len(self.inputs)
        hid_layers = self.hidden
        num_out = len(self.outputs)
        
        for i in range(len(hid_layers)):
            num_hid = len(hid_layers[i])
                
            if i == 0:
                for j in range(num_in):
                    for k in range(1, num_hid):
                        syn = Synapse(self.inputs[j], self.hidden[i][k], weights[index])
                        self.inputs[j].synapses.append(syn)
                        self.hidden[i][k].synapses.append(syn)
                        index += 1
                        
                if len(hid_layers) == 1:
                    for j in range(num_hid):
                        for k in range(num_out):
                            syn = Synapse(self.hidden[i][j], self.outputs[k], weights[index])
                            self.hidden[i][j].synapses.append(syn)
                            self.outputs[k].synapses.append(syn)
                            index += 1
                continue
            
            for j in range(len(hid_layers[i-1])):
                for k in range(1, num_hid):
                    syn = Synapse(self.hidden[i-1][j], self.hidden[i][k], weights[index])
                    self.hidden[i-1][j].synapses.append(syn)
                    self.hidden[i][k].synapses.append(syn)
                    index += 1
                    
            if i == len(hid_layers) - 1:
                for j in range(num_hid):
                    for k in range(num_out):
                        syn = Synapse(self.hidden[i][j], self.outputs[k], weights[index])
                        self.hidden[i][j].synapses.append(syn)
                        self.outputs[k].synapses.append(syn)
                        index += 1
                break

    def initial_weights(self):
        num_in = len(self.inputs)
        layers = self.hidden
        num_out = len(self.outputs)
        
        num_wgt = 0
        for i in range(len(layers)):
            if i == 0:
                num_wgt += (num_in + 1) * len(layers[0])
                continue
                
            num_wgt += (len(layers[i-1]) + 1) * len(layers[i])
            
            if i == len(layers) - 1:
                num_wgt += (len(layers[i]) + 1) * num_out
                break
        
        weights = []
        for i in range(num_wgt):
            weights.append(round(random.uniform(-1, 1), 2))
            
        return weights

    def sanitize(self, num_in, layers, num_out, funcs, weights):
        if len(layers) < 1:
            raise Exception("Invalid number of hidden layer neurons. Must be at least one.")
        
        num_wgt = 0
        for i in range(len(layers)):                
            if i == 0:
                num_wgt += (num_in + 1) * layers[0]
                
                if len(layers) == 1:
                    num_wgt += (layers[i] + 1) * num_out
                continue
                
            num_wgt += (layers[i-1] + 1) * layers[i]
            
            if i == len(layers) - 1:
                num_wgt += (layers[i] + 1) * num_out
                break
            
        if num_in < 1:
            raise Exception("Invalid number of input neurons. Must be at least one.")
        elif num_out < 1:
            raise Exception("Invalid number of output neurons. Must be at least one.")
        elif len(funcs) != len(layers)+1:
            raise Exception("Invalid number of activation functions. Expected: {} but got: {}".format(len(layers)+1, len(funcs)))
        elif (weights is not None) and len(weights) != num_wgt:
            raise Exception("Invalid number of weights. Expected: {} but got: {}".format(num_wgt, len(weights)))
        
    def __repr__(self):
        string = "\n- Multi-Layer Perceptron -"
        string += "\n-------------------------------------------------------"
        
        string += "\n\n  - Input"
        string += "\n    | Neurons"
        for n in self.inputs:
            string += repr(n)
            
        l = 1
        string += "\n\n  - Hidden Layers"
        for layer in self.hidden:
            string += "\n    | Neurons - L"+str(l)
            l += 1
            for n in layer:
                string += repr(n)
                
        string += "\n\n  - Output"
        string += "\n    | Neurons"
        for n in self.outputs:
            string += repr(n)
            
        string += "\n\n-------------------------------------------------------"
        string += "\n  - Alpha: "+str(self.alpha)
        
        return string
