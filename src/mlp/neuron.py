#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Neuron(object):
    
    def __init__(self, idn, layer, value=0.0, func="N/A"):
        self.id = idn
        self.layer = layer
        self.synapses = []
        self.value = value
        self.function = func
        self.net = None
        self.sensibility = None
    
    def evaluate(self):
        if self.function == "Linear":
            self.value = self.linear()
            return self.value
        
        if self.function == "Sigmoid":
            self.value = self.sigmoid()
            return self.value
        
    def backpropagate(self, error=None, layers=3):
        if error:
            if self.function == "Linear":
                self.sensibility = error
            
            if self.function == "Sigmoid":
                self.sensibility = self.sigmoid_deriv() * error
            
            return
        
        sensibility = 0
        for syn in self.synapses:
            if syn.end.layer == str(layers):
                sensibility += syn.end.sensibility * syn.weight
        
        if self.function == "Linear":
            self.sensibility = sensibility
        
        if self.function == "Sigmoid":
            self.sensibility = self.sigmoid_deriv() * sensibility
            
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
