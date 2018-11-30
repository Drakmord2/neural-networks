#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .neuron import Neuron
from .synapse import Synapse
import random

 
class MLP(object):
    """
    Multi-Layer Perceptron
    
    Parameters: alpha=Learning constant; numIn=Neurons on input layer;
                numHid=Neurons on hidden layer; numOut=Neurons on output layer;
                funcs=Tuple with activation function of each layer;
                weights=List with all synaptic weights;
    """
    
    def __init__(self, alpha, num_in, num_hid, num_out, funcs, weights):
        """
        Build a network
        """
        self.alpha = alpha
        self.inputs = []
        self.hidden = []
        self.outputs = []
        self.errors = []
        self.mse = -1.0
        
        self.sanitize(num_in, num_hid, num_out, funcs, weights)
        self.init_neurons(num_in, num_hid, num_out, funcs, weights)

    def train(self, example):
        """
        Network training
        """
        for neuron in self.inputs[1:]:
            neuron.value = example["q"]
        
        error = self.feedforward(example["pq"])
        self.feedback(error)
        self.learning()
        self.errors.append(round(error, 4))
        
    def compute(inputs):
        """
        Given a set of inputs, Classify or Predict
        """
        pass
        
    def feedforward(self, expected):
        """
        feedForward algorithm
        
        Parameters: expected=expected output value
        """
        error = 0
        
        for layer in range(len(self.hidden)):
            for neuron in self.hidden[layer][1:]:    
                net = 0
                for syn in neuron.synapses:
                    if syn.end == neuron:
                        net += syn.start.value * syn.weight
                neuron.net = net
                neuron.evaluate()
            
        for neuron in self.outputs:
            net = 0
            for syn in neuron.synapses:
                if syn.end == neuron:
                    net += syn.start.value * syn.weight
            neuron.net = net
            neuron.evaluate()
            
            error = expected - neuron.value #TODO Generalize
        
        return error
    
    def feedback(self, error):
        """
        Backpropagation algorithm
        
        Parameters: error=expect output minus derived output
        """
        for neuron in self.outputs:
            neuron.backpropagate(error)
        
        for layer in range(len(self.hidden)):
            for neuron in self.hidden[layer][1:]: 
                neuron.backpropagate(layers=len(self.hidden)+2)
    
    def learning(self):
        """
        Learning algorithm
        
        W(new) = W(old) + alpha + sensibility + f(net)
        """
        for layer in range(len(self.hidden)):
            for neuron in self.hidden[layer]:
                for syn in neuron.synapses:
                    if syn.start == neuron:
                        syn.weight = syn.weight + (self.alpha * syn.end.sensibility * neuron.value) 
                        
        for neuron in self.inputs:
            for syn in neuron.synapses:
                if syn.start == neuron:
                    syn.weight = syn.weight + (self.alpha * syn.end.sensibility * neuron.value)        

    def get_mse(self):
        """
        Mean Square Error
        
        MSE = 1/N * ∑ e^2
        """
        mse = 0
        for error in self.errors:
            mse += error ** 2
        mse = mse / len(self.errors)
        
        self.mse = mse

    def get_optimal_weights(self):
        weights = []
        for neuron in self.hidden:
            for syn in neuron.synapses:
                if syn.start == neuron:
                    weights.append(syn.weight)
        
        for neuron in self.inputs:
            for syn in neuron.synapses:
                if syn.start == neuron:
                    weights.append(syn.weight)
                   
        return weights

    def init_neurons(self, num_in, num_hid, num_out, funcs, weights):
        # Threshold neurons
        neuron = Neuron('0', '1', 1.0)
        self.inputs.append(neuron)
        
        for n in range(0, len(num_hid)):
            neuron = Neuron('0', str(n+2), 1.0)
            self.hidden.append([neuron])
        
        # Neurons
        for n in range(num_in):
            idn = str(n+1)
            neuron = Neuron(idn, '1')
            
            self.inputs.append(neuron)
        
        for l in range(0, len(num_hid)):
            neurons = []
            for n in range(num_hid[l]):
                idn = str(n+1)
                neuron = Neuron(idn, str(l+2), func=funcs[l])
                neurons.append(neuron)
                
            self.hidden[l] += neurons

        for n in range(num_out):
            idn = str(n+1)
            neuron = Neuron(idn, str(len(num_hid)+2), func=funcs[len(num_hid)])
            
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
        
        for i in range(0, len(hid_layers)):
            num_hid = len(hid_layers[i])
            
            if i == 0:
                for j in range(num_in):
                    for k in range(1, num_hid):
                        syn = Synapse(self.inputs[j], self.hidden[i][k], weights[index])
                        self.inputs[j].synapses.append(syn)
                        self.hidden[i][k].synapses.append(syn)
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
        num_hid = self.hidden
        num_out = len(self.outputs)
        
        num_wgt = 0
        for i in range(0, len(num_hid)):
            if i == 0:
                num_wgt += (num_in + 1) * num_hid[0]
                continue
                
            num_wgt += (num_hid[i-1] + 1) * num_hid[i]
            
            if i == len(num_hid) - 1:
                num_wgt += (num_hid[i] + 1) * num_out
                break
        
        weights = []
        for i in range(num_wgt):
            weights.append(round(random.uniform(-1, 1), 2))
            
        return weights

    def sanitize(self, num_in, num_hid, num_out, funcs, weights):
        if len(num_hid) < 1:
            raise Exception("Invalid number of hidden layer neurons")
            
        num_wgt = 0
        for i in range(0, len(num_hid)):
            if i == 0:
                num_wgt += (num_in + 1) * num_hid[0]
                continue
                
            num_wgt += (num_hid[i-1] + 1) * num_hid[i]
            
            if i == len(num_hid) - 1:
                num_wgt += (num_hid[i] + 1) * num_out
                break
            
        if num_in < 1:
            raise Exception("Invalid number of input neurons")
        elif num_out < 1:
            raise Exception("Invalid number of output neurons")
        elif len(funcs) != len(num_hid)+1:
            raise Exception("Invalid number of activation functions")
        elif (weights is not None) and len(weights) != num_wgt:
            raise Exception("Invalid number of weights")
        
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
        string += "\n  - MSE: "+str(self.mse)
        
        return string
