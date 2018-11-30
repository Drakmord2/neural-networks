#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
if getattr(sys, 'ps1', sys.flags.interactive):
    from mlp.mlp import MLP
    from visualization.visualization import Visualization
else:
    from src.mlp.mlp import MLP
    from src.visualization.visualization import Visualization

# Data
training = [
        {'q': 2.5, 'pq': 0.18},
        {'q': 3.5, 'pq': 0.65},
        {'q': 4.5, 'pq': 0.96},
        {'q': 1.5, 'pq': 0.02},
        {'q': 3.0, 'pq': 0.50},
        {'q': 4.0, 'pq': 0.88},
        {'q': 2.0, 'pq': 0.08}
        ]
cross_validation = []
test = []

if __name__ == "__main__":
    layer_function = [               # Activation Function of each layer
            "Sigmoid", "Sigmoid", "Sigmoid",
            "Sigmoid"
            ] 
    num_in = 1                      # Number of input layer neurons
    num_hid = [2, 1, 2]             # Number of neurons in each hidden layer
    num_out = 1                     # Number of output layer neurons
                                    # * Threshold neurons added automatically *
    alpha = 0.86                    # Learning rate
    weights = [                     # Synaptic weights 
            0.3, -0.7, 0.4, 0.5,
            -0.6, 0.1, -0.8,
            -0.4, 0.2, 0.3, -0.3,
            0.3, -0.5, 0.1
            ] 
    cycles = 15                     # Training cycles
    
    try:
        network = MLP(alpha, num_in, num_hid, num_out, layer_function, weights)
        mses = []
        
        for cycle in range(cycles):
            network.errors = []
            for example in training:
                network.train(example)
            network.get_mse()
            mses.append(network.mse)
        
        print(network)
        print("  - "+str(cycles)+" Cycles")
        print("  - "+str(len(training))+" Examples")
        print("-------------------------------------------------------")
        
        vz = Visualization()
        print("\n- Statistics -")
        vz.learning_curve(mses)
        print("\n- Architecture -")
        vz.network(network)
        print("\n===========================================================\n")
    except Exception as err:
        print("\n------\nError: {0}\n------".format(err))
