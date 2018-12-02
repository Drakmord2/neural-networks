#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Neuron(object):
    
    def __init__(self, idn, layer, value=0.0, func="N/A"):
        """
        Build a Neuron
        """
        self.id = idn               # String
        self.layer = layer          # String
        self.synapses = []          # List(Synapse)
        self.value = value          # Float
        self.function = func        # String
        self.net = None             # Float
        self.sensitivity = None     # Float
    
    def evaluate(self):
        """
        Calculate output
        """
        if self.function == "Linear":
            self.value = self.linear()
            return self.value
        
        if self.function == "Sigmoid":
            self.value = self.sigmoid()
            return self.value
        
    def backpropagate(self, error=None):
        """
        Calculate sensitivity
        """
        if error:
            if self.function == "Linear":
                self.sensitivity = error
            
            if self.function == "Sigmoid":
                self.sensitivity = self.sigmoid_deriv() * error
            
            return
        
        sensitivity = 0
        for syn in self.synapses:
            if syn.start == self:
                sensitivity += syn.end.sensitivity * syn.weight
        
        if self.function == "Step":
            self.sensitivity = 0
            
        if self.function == "Linear":
            self.sensitivity = sensitivity
        
        if self.function == "Sigmoid":
            self.sensitivity = self.sigmoid_deriv() * sensitivity
      
    def step(self):
        if self.net > 0:
            return 1
        
        return 0
    
    def linear(self):
        y = self.net
        return y
    
    def sigmoid(self):
        y = 1 / (1 + np.exp(-self.net))
        return y
    
    def sigmoid_deriv(self):
        y = self.value
        dy = y * (1 - y)
        return dy
    
    def __repr__(self):
        string = ""
        
        string += "\n      | id: "+str(self.id)
        string += "\n      | value: "+str(self.value)
        string += "\n      | function: "+self.function
        string += "\n      + Synapses"
        for s in self.synapses:
            string += s.to_string(self)
        
        return string
