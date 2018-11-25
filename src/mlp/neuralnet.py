#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mlp import MLP
import matplotlib.pyplot as plt

training = [
        {'q': 2.5, 'pq': 0.18},
        {'q': 3.5, 'pq': 0.65},
        {'q': 4.5, 'pq': 0.96},
        {'q': 1.5, 'pq': 0.02},
        {'q': 3.0, 'pq': 0.50},
        {'q': 4.0, 'pq': 0.88},
        {'q': 2.0, 'pq': 0.08}
        ]
crossValidation = []
test = []

if __name__ == "__main__":
    layerFunction = ("Sigmoid", "Sigmoid")
    numIn = 1
    numHid = 2
    numOut = 1
    alpha = 0.7
    weights = [0.1, -0.7, -0.3, 0.4, -0.6, 0.1, -0.8]
    
    try:
        network = MLP(alpha, numIn, numHid, numOut, layerFunction, weights)
        mses = []
        cycles = 15
        
        for cycle in range(cycles):
            network.errors = []
            for example in training:
                network.train(example)
            network.get_mse()
            mses.append(network.mse)
        
        print(network)
        print("  - "+str(cycles)+" Cycles")
        print("  - "+str(len(training))+" Examples")
        print("-----------------------------------------------------")
        
        plt.title("Learning Curve")
        plt.plot(mses)
        plt.show()
    except Exception as err:
        print("\n------\nError: {0}\n------".format(err))
