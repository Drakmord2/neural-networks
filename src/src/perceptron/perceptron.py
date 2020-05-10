#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..mlp.neuron import Neuron
from ..mlp.synapse import Synapse
import random
import math


class Perceptron(object):
    """
    Perceptron Neural Network
    
    Parameters: alpha=Learning rate; numIn=Neurons on input layer;
                numOut=Neurons on output layer;
                funcs=Tuple with activation function of each layer;
                weights=List with all synaptic weights;
    """

    def __init__(self, alpha, num_in, num_out, funcs, weights, loss_func):
        """
        Build a network
        """
        self.type = "Perceptron"
        self.alpha = alpha
        self.inputs = []
        self.outputs = []
        self.errors = []
        self.loss_func = loss_func
        self.loss = -1.0

        self.sanitize(num_in, num_out, funcs, weights)
        self.init_neurons(num_in, num_out, funcs, weights)

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
        if (len(self.inputs) - 1 != len(example) - outputs):
            raise Exception("Invalid input data format")

        self.get_inputs(example)

        expected = example.iloc[len(example) - outputs:]

        error = self.feedforward(expected)
        self.learning(expected)
        self.errors += error

    def feedforward(self, expected=None):
        """
        feedForward algorithm
        
        Parameters: expected=expected output values
        """

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
                error.append((expected[n], float(neuron.value)))
            n += 1

        return error

    def learning(self, expected):
        """
        Learning algorithm
        
        W(new) = W(old) + alpha * (Di - Yi) * Xj
        """
        for neuron in self.inputs:
            for syn in neuron.synapses:
                if syn.start == neuron:
                    syn.weight = syn.weight + (self.alpha * (expected[int(syn.end.id) - 1] - syn.end.value) * neuron.value)

    def get_inputs(self, data):
        for i in range(1, len(self.inputs)):
            neuron = self.inputs[i]
            neuron.value = data.iloc[i - 1]

    def get_loss(self):
        if self.loss_func == 'MSE':
            return self.get_mse()

        if self.loss_func == 'CE':
            return self.get_ce()

    def get_ce(self):
        """
        Cross-Entropy
        
        CE = - 1 / N ∑ d * log(y) + (1 - d) * log(1 - y)
        """
        cea = 0
        for output in self.errors:
            eps = 1e-15
            predicted = max(min(output[1], 1 - eps), eps)

            cea += output[0] * math.log(predicted) + (1 - output[0]) * math.log(1 - predicted)

        ce = - cea / len(self.errors)

        self.loss = ce

        return ce

    def get_mse(self):
        """
        Mean Squared Error
        
        MSE = 1 / N ∑ [ (d - y)^2 ]
        """
        mse = 0
        for error in self.errors:
            mse += (error[0] - error[1]) ** 2
        mse = mse / len(self.errors)

        self.loss = mse

        return mse

    def get_optimal_weights(self):
        weights = []

        for neuron in self.inputs:
            for syn in neuron.synapses:
                if syn.start == neuron:
                    weights.append(float(syn.weight))

        return weights

    def get_output(self):
        outputs = []
        for out in self.outputs:
            outputs.append(out.value)

        return outputs

    def init_neurons(self, num_in, num_out, funcs, weights):
        # Threshold neurons
        neuron = Neuron('0', '1', 1.0)  # Input
        self.inputs.append(neuron)

        # Neurons
        for n in range(num_in):  # Input
            idn = str(n + 1)
            neuron = Neuron(idn, '1')

            self.inputs.append(neuron)

        for n in range(num_out):  # Output
            idn = str(n + 1)
            neuron = Neuron(idn, '2', func=funcs[0])

            self.outputs.append(neuron)

        # Connections
        if weights:
            self.init_synapses(weights)
        else:
            self.init_synapses(self.initial_weights())

    def init_synapses(self, weights):
        index = 0
        num_in = len(self.inputs)
        num_out = len(self.outputs)

        for i in range(num_in):
            for j in range(num_out):
                syn = Synapse(self.inputs[i], self.outputs[j], weights[index])
                self.inputs[i].synapses.append(syn)
                self.outputs[j].synapses.append(syn)
                index += 1

    def initial_weights(self):
        num_in = len(self.inputs)
        num_out = len(self.outputs)
        num_wgt = (num_in + 1) * num_out

        weights = []
        for i in range(num_wgt):
            weights.append(round(random.uniform(-1, 1), 2))

        return weights

    def sanitize(self, num_in, num_out, funcs, weights):
        num_wgt = (num_in + 1) * num_out

        if num_in < 1:
            raise Exception("Invalid number of input neurons. Must be at least one.")
        elif num_out < 1:
            raise Exception("Invalid number of output neurons. Must be at least one.")
        elif len(funcs) != 1:
            raise Exception("Invalid number of activation functions. Expected: {} but got: {}".format(1, len(funcs)))
        elif (weights is not None) and len(weights) != num_wgt:
            raise Exception("Invalid number of weights. Expected: {} but got: {}".format(num_wgt, len(weights)))

    def __repr__(self):
        string = "\n- Perceptron -"
        string += "\n-------------------------------------------------------"

        string += "\n\n  - Input"
        string += "\n    | Neurons"
        for n in self.inputs:
            string += repr(n)

        string += "\n\n  - Output"
        string += "\n    | Neurons"
        for n in self.outputs:
            string += repr(n)

        string += "\n\n-------------------------------------------------------"
        string += "\n  - Alpha: " + str(self.alpha)

        return string
