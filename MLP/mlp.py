#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from neuron import Neuron
from synapse import Synapse
import random

 
class MLP(object):
    """
    Multi-Layer Perceptron
    
    Parameters: alpha=Learning constant; numIn=Neurons on input layer;
                numHid=Neurons on hidden layer; numOut=Neurons on output layer;
                funcs=Tuple with activation function of each layer;
                weights=List with all synaptic weights;
    """
    
    def __init__(self, alpha, numIn=1, numHid=1, numOut=1, funcs=[], weights=None):
        self.alpha = alpha
        self.inputs = []
        self.hidden = []
        self.outputs = []
        self.errors = []
        self.mse = -1.0
        
        self.sanitize(numIn, numHid, numOut, funcs, weights)
        self.initNeurons(numIn, numHid, numOut, funcs, weights)

    def feedForward(self, expected):
        """
        feedForward algorithm
        
        Parameters: expected=expected output value
        """
        error = 0
        
        for neuron in self.hidden[1:]:    
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
            
            error = expected - neuron.value
        
        return error
    
    def feedBack(self, error):
        """
        Backpropagation algorithm
        
        Parameters: error=expect output minus derived output
        """
        for neuron in self.outputs:
            neuron.backpropagate(error)
        
        for neuron in self.hidden[1:]: 
            neuron.backpropagate()
    
    def learning(self):
        """
        Learning algorithm
        
        W(new) = W(old) + alpha + sensibility + f(net)
        """
        for neuron in self.hidden:
            for syn in neuron.synapses:
                if syn.start == neuron:
                    syn.weight = syn.weight + (self.alpha * syn.end.sensibility * neuron.value) 
        for neuron in self.inputs:
            for syn in neuron.synapses:
                if syn.start == neuron:
                    syn.weight = syn.weight + (self.alpha * syn.end.sensibility * neuron.value)        
            
    def train(self, example):
        """
        Network training
        """
        for neuron in self.inputs[1:]:
            neuron.value = example["q"]
        
        error = self.feedForward(example["pq"])
        self.feedBack(error)
        self.learning()
        self.errors.append(round(error,4))

    def getMSE(self):
        """
        Mean Square Error
        
        MSE = 1/N * ∑ e^2
        """
        mse = 0
        for error in self.errors:
            mse += error**2
        mse = mse / len(self.errors)
        
        self.mse = mse

    def getOptimalWeights(self):
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

    def initNeurons(self, numIn, numHid, numOut, funcs, weights):            
        # Threshold neurons
        neuron = Neuron('0', '1', 1.0)
        self.inputs.append(neuron)
        neuron = Neuron('0', '2', 1.0)
        self.hidden.append(neuron)
        
        for n in range(numIn):
            id = str(n+1)
            neuron = Neuron(id, '1')
            
            self.inputs.append(neuron)
        
        for n in range(numHid):
            id = str(n+1)
            neuron = Neuron(id, '2', function=funcs[0])
            
            self.hidden.append(neuron) 

        for n in range(numOut):
            id = str(n+1)
            neuron = Neuron(id, '3', function=funcs[1])
            
            self.outputs.append(neuron)
            
        if weights:
            self.initSynapses(weights)
        else:
            self.initSynapses(self.initialWeights())

    def initSynapses(self, weights):
        index = 0
        numIn = len(self.inputs)
        numHid = len(self.hidden)
        numOut = len(self.outputs)
        
        for i in range(numIn):
            for j in range(1, numHid):
                syn = Synapse(self.inputs[i], self.hidden[j], weights[index])
                self.inputs[i].synapses.append(syn)
                self.hidden[j].synapses.append(syn)
                index += 1
                
        for i in range(numHid):
            for j in range(numOut):
                syn = Synapse(self.hidden[i], self.outputs[j], weights[index])
                self.hidden[i].synapses.append(syn)
                self.outputs[j].synapses.append(syn)
                index += 1

    def initialWeights(self):
        numIn = len(self.inputs)
        numHid = len(self.hidden)
        numOut = len(self.outputs)
        numWgt = (numIn * (numHid - 1)) + (numHid * numOut)
        
        weights = []
        for i in range(numWgt):
            weights.append(round(random.uniform(-1,1), 2))
            
        return weights

    def sanitize(self, numIn, numHid, numOut, funcs, weights):
        numWgt = ((numIn+1) * numHid) + ((numHid+1) * numOut)
        
        if numIn < 1:
            raise Exception("Invalid number of input neurons")
        elif numHid < 1:
            raise Exception("Invalid number of hidden layer neurons")
        elif numOut < 1:
            raise Exception("Invalid number of output neurons")
        elif len(funcs) != 2:
            raise Exception("Invalid number of activation functions")
        elif (weights is not None) and len(weights) != numWgt:
            raise Exception("Invalid number of weights")
        
    def __repr__(self):
        string = "\n- Multi-Layer Perceptron -"
        string += "\n-----------------------------------------------------"
        
        string += "\n\n  - Input"
        string += "\n    | Neurons"
        for n in self.inputs:
            string += repr(n)
        string += "\n\n  - Hidden Layer"
        string += "\n    | Neurons"
        for n in self.hidden:
            string += repr(n)
        string += "\n\n  - Output"
        string += "\n    | Neurons"
        for n in self.outputs:
            string += repr(n)
            
        string += "\n\n-----------------------------------------------------"
        string += "\n  - Alpha: "+str(self.alpha)
        string += "\n  - MSE: "+str(self.mse)
        
        return string
