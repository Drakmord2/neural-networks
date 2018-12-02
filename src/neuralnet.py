#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
if getattr(sys, 'ps1', sys.flags.interactive):
    from mlp.mlp import MLP
    from visualization.visualization import Visualization
    from openai.openai import OpenAI
else:
    from src.mlp.mlp import MLP
    from src.visualization.visualization import Visualization
    from src.openai.openai import OpenAI

def getData(path=None):
    """
        Fetches data from a CSV file.
        
        Format: Inputs followed by outputs
    """
    data = pd.read_csv(path)
    data = data.sample(frac=1)
    
    size = round(len(data)*0.8)
    
    training = data[:size]
    cross_validation = []
    test = data[size:]
    
    return training, cross_validation, test

def show_result(network, cycles=None, training=None, mses=None, output=None):
    print(network)
    
    if output:
        print("  - Outputs: "+str(output))
        
    if cycles:
        print("  - "+str(cycles)+" Cycles")
        
    if training is not None:
        print("  - "+str(len(training))+" Examples")
    print("-------------------------------------------------------")
    
    vz = Visualization()
    
    if mses:
        print("\n- Statistics -")
        vz.learning_curve(mses)
        
    print("\n- Architecture -")
    vz.network(network)
    print("\n=========================================================\n")
    print("Optimal Weights: ", network.get_optimal_weights())

def setup():
    layer_function = [
        "Sigmoid", "Sigmoid",
        "Sigmoid"
        ]                           # Activation Function of each layer
    num_in = 1                      # Number of input layer neurons
    num_hid = [2, 3]                # Number of neurons in each hidden layer
    num_out = 1                     # Number of output layer neurons
                                    # * Threshold neurons added automatically *
    alpha = 0.82                    # Learning rate
    weights = [                     # Synaptic weights 
            0.3, -0.7, 0.4, 0.5,
            -0.6, 0.2 ,0.1, -0.3, -0.8, 0.9, 0.8, 0.6, 0.5,
            0.3, -0.5, 0.1, -0.6
            ] 
    
    network = MLP(alpha, num_in, num_hid, num_out, layer_function, weights)

    return network

def training(network, data):
    outputs = 1
    mses = []
    cycles = 2500
    
    for cycle in range(cycles):
        network.errors = []
    
        for _, example in data.iterrows():
            network.train(example, outputs)
            
        mses.append(network.get_mse())
    
    return mses, cycles

def setup_openai():
    oai = OpenAI()
    oai.hotterColder()
    

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    data_path = "data/data1.csv"
    training_data,_,test_data = getData(data_path)
    
    try:
        network = setup()

        # Training
        mses, cycles = training(network, training_data)
        show_result(network, cycles, training_data, mses)
        
        # Computing
        inputs = list(test_data.iterrows())
        print("\n- Tests\n")
        for i in range(len(inputs)):
            inp = inputs[i][1]
            
            result = network.compute(inp)
            print("  - Result: {} | Expected: {}".format(result, inp[1]))
        
    except Exception as err:
        print("\n------\nError: {0}\n------".format(err))
