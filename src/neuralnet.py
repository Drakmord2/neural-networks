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


# =============================================================================
# Globals
# =============================================================================

CV_THRESH = 250
MAX_CYCLES = 2000

# =============================================================================
# Auxiliary Functions
# =============================================================================
    
def getCSVData(path=None, sep=',', decimal='.'):
    """
        Fetches data from a CSV file.
        
        Format: Inputs followed by outputs
    """
    data = pd.read_csv(path, sep=sep, decimal=decimal)
    data = data.sample(frac=1)
    
    size = round(len(data)*0.7)
    training = data[:size]
    
    data = data[size:]
    size = round(len(data)*0.66)
    
    cross_validation = data[:size]
    test = data[size:]
    
    return training, cross_validation, test

def show_result(network, cycles=None, training=None, mses=None, cverr=None, output=None):
    print(network)
    
    if output:
        print("  - Outputs: "+str(output))
        
    if cycles:
        print("  - "+str(cycles)+" Cycles")
        
    if training is not None:
        print("  - "+str(len(training))+" Training Examples")
    print("-------------------------------------------------------")
    
    vz = Visualization()
    
    if mses:
        print("\n- Statistics -")
        vz.learning_curve(mses, cverr)
        
    print("\n- Architecture -")
    vz.network(network)
    print("\n=========================================================\n")
    print("Optimal Weights: ", network.get_optimal_weights())

def setup(num_in, num_out, binary=False):
    layer_function = [
        "Sigmoid",
        "Sigmoid"
        ]                           # Activation Function of each layer
    num_in = num_in                 # Number of input layer neurons
    num_hid = [2]                   # Number of neurons in each hidden layer
    num_out = num_out               # Number of output layer neurons
    alpha = 0.05                    # Learning rate
    weights = [                     # Synaptic weights 
            0.3, -0.7, 0.4, 0.5,
            0.3, -0.5, 0.1
            ] 
    weights = None
    loss_function = "CE" if binary else "MSE"
    
    network = MLP(alpha, num_in, num_hid, num_out, layer_function, weights, loss_function)

    return network

def training(network, training, validation, num_in=1, num_out=1):
    mses = []
    cvmse = []
    cycles = 0
    
    while True:
        network.errors = []
    
        for _, example in training.iterrows():
            network.train(example, num_out)
            
        mse = network.get_loss()
        mses.append(mse)
        
        cvmse, cverr = cross_validation(network, validation, cvmse, num_in)
        
        if cycles > CV_THRESH:
            if mse < mses[cycles-1] and cverr >= cvmse[cycles-1] and cverr >= mse:
                break
        
        cycles += 1
        
        if cycles >= MAX_CYCLES:
            break
        
    return mses, cvmse, cycles

def cross_validation(network, validation, cvmse, num_in=1):
    err = []
    for _, example in validation.iterrows():
        result = network.compute(example)
        err.append(example[num_in] - result)
    
    cverr = 0
    for error in err:
        cverr += error ** 2
    cverr = cverr / len(err)
    cvmse.append(cverr)
    
    return cvmse, cverr

def compute(network, data, num_in=1, num_out=1):
    inputs = data.iloc[:,:-num_out]
    output = data.iloc[:,-num_out:]
    print("\n- Tests\n")
    
    err = []
    i = 0
    for idx, row in inputs.iterrows():
        result = network.compute(row)
        print("  - Result: {} | Expected: {}".format(result[0], output.iloc[i][0]))
        err.append(result[0] - output.iloc[i][0])
        i += 1
        
    mse = 0
    for error in err:
        mse += error ** 2
    mse = mse / len(err)
    print("  - {}: {}".format(network.loss_func, mse))
    
def setup_openai():
    oai = OpenAI()
    oai.hotterColder()
    

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    try:
        # Data import
        data_path = "data/data1.csv"
        training_data, validation_data, test_data = getCSVData(data_path, sep=',', decimal='.')
        
        # Data structure
        num_in = 1
        num_out = 1
        binary_output = False
        
        # Network setup
        network = setup(num_in, num_out, binary_output)

        # Training
        mses, cverr, cycles = training(network, training_data, validation_data, num_in, num_out)
        show_result(network, cycles, training_data, mses, cverr)
        
        # Computing
        compute(network, test_data, num_in, num_out)
        
    except Exception as err:
        print("\n------\nError: {0}\n------".format(err))
