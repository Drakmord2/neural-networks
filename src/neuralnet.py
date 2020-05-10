#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import math
from src.mlp.mlp import MLP
from src.perceptron.perceptron import Perceptron
from src.visualization.visualization import Visualization
from src.openai.openai import OpenAI

# =============================================================================
# Globals
# =============================================================================

K_FOLDS = 3
CV_THRESH = 50
MAX_CYCLES = 500


# =============================================================================
# Auxiliary Functions
# =============================================================================


def getCSVData(path=None, sep=',', decimal='.', share=0.7):
    """
        Fetches data from a CSV file.
        
        Format: Inputs followed by outputs
    """
    data = pd.read_csv(path, sep=sep, decimal=decimal)  # Read file
    data = data.sample(frac=1)  # Shuffle data

    size = round(len(data) * share)  # Pick 90% of data to Training/Validation set
    train_val = data[:size]
    test = data[size:]  # 10% to Test set

    return train_val, test


def show_result(network, cycles=None, training=None, loss=None, cverr=None, output=None):
    print(network)

    if output:
        print("  - Outputs: " + str(output))

    if cycles:
        print("  - " + str(cycles) + " Cycles")

    if training is not None:
        print("  - " + str(len(training)) + " Training Examples")
    print("-------------------------------------------------------")

    vz = Visualization()

    if loss:
        print("\n- Statistics -")
        vz.learning_curve(loss, cverr, network.loss_func)

    print("\n- Architecture -")
    vz.network(network)
    print("\n=========================================================\n")
    print("- Optimal Weights\n    ", network.get_optimal_weights())


def setup(arch, num_in, num_out, num_hid=[1], layer_function=["Linear", "Sigmoid"], weights=None, alpha=0.7, binary=False):
    """
        Create network
        
    Parameters:
        arch            = Neural Network Architecture
        layer_function  = Activation Function of each layer
        num_in          = Number of input layer neurons
        num_hid         = Number of neurons in each hidden layer
        num_out         = Number of output layer neurons
        alpha           = Learning rate
        weights         = Synaptic weights 
        binary          = Binary output
    """
    loss_function = "CE" if binary else "MSE"

    network = None

    if arch == "MLP":
        network = MLP(alpha, num_in, num_hid, num_out, layer_function, weights, loss_function)
    if arch == "Perceptron":
        network = Perceptron(alpha, num_in, num_out, layer_function, weights, loss_function)

    return network


def training(network, data, num_in=1, num_out=1):
    losses = []
    cvloss = []
    cycles = 0
    curr_k = 1
    print("\n- Training")
    training, validation = kfold(data, curr_k)

    while True:
        network.errors = []

        for _, example in training.iterrows():
            network.train(example, num_out)

        loss = network.get_loss()
        losses.append(loss)

        cvloss, cverr = cross_validation(network, validation, cvloss, num_in)

        if cycles > CV_THRESH:
            if loss < losses[cycles - 1] and cverr >= losses[cycles - 1] and cverr >= loss:
                break

        if cycles % 10 == 0:
            print("    Cycle: {}".format(cycles + 10))
        cycles += 1

        if cycles >= MAX_CYCLES:
            break

    return losses, cvloss, cycles


def cross_validation(network, validation, cvmse, num_in=1):
    err = []
    for _, example in validation.iterrows():
        result = network.compute(example)
        err.append((example[num_in], float(result[0])))

    if network.loss_func == "MSE":
        cverr = get_mse(err)
    if network.loss_func == "CE":
        cverr = get_ce(err)

    cvmse.append(cverr)

    return cvmse, cverr


def kfold(data, curr_k):
    training = data  # [100:]
    validation = data  # [:100]
    return training, validation


def compute(network, data, num_in=1, num_out=1):
    inputs = data.iloc[:, :-num_out]  # Excludes output data
    output = data.iloc[:, -num_out:]  # Excludes input data
    print("\n- Tests")

    err = []
    i = 0
    hit = 0
    count = 0
    for idx, row in inputs.iterrows():
        result = network.compute(row)

        if network.loss_func == "CE":
            if result == list(output.iloc[i]):
                hit += 1
            count += 1

        for j in range(len(result)):
            err.append((int(result[j]), output.iloc[i][j]))
        i += 1

    if network.loss_func == "CE":
        print("  - Hits: {} | Total: {}\n  - Accuracy: {}".format(hit, count, hit / count))
        loss = get_ce(err)

    if network.loss_func == "MSE":
        loss = get_mse(err)

    print("  - {}: {}".format(network.loss_func, loss))


def get_ce(err):
    """
    Cross-Entropy
    
    - 1 / N ∑ d * log(y) + (1 - d) * log(1 - y)
    """
    cea = 0
    for output in err:
        eps = 1e-15
        predicted = max(min(output[1], 1 - eps), eps)

        cea += output[0] * math.log(predicted) + (1 - output[0]) * math.log(1 - predicted)

    ce = - cea / len(err)

    return ce


def get_mse(err):
    """
    Mean Squared Error
    
    1 / N ∑ [ (d - y)^2 ]
    """
    mse = 0
    for error in err:
        mse += (error[0] - error[1]) ** 2
    mse = mse / len(err)

    return mse


# =============================================================================
# Main
# =============================================================================

def mlp():
    # Data import
    data_path = "data/data1.csv"
    training_data, test_data = getCSVData(data_path, sep=',', decimal='.')

    # Data structure
    num_in = 1
    num_out = 1
    binary_out = False

    # Network parameters
    layer_function = ["Linear", "Sigmoid"]
    hidden = [2]
    weights = [0.3, -0.7, 0.4, 0.5, 0.3, -0.5, 0.1]
    alpha = 0.7

    # Network setup
    network = setup("MLP", num_in, num_out, hidden, layer_function, weights, alpha, binary_out)

    # Training
    mses, cverr, cycles = training(network, training_data, num_in, num_out)
    show_result(network, cycles, training_data, mses, cverr)

    # Computing
    compute(network, test_data, num_in, num_out)


def perceptron():
    # Data import
    data_path = "data/cartpole_data.csv"
    training_data, test_data = getCSVData(data_path, sep=',', decimal='.')

    # Data structure
    num_in = 4
    num_out = 1
    binary_out = True

    # Network parameters
    layer_function = ["Step"]
    weights = None
    alpha = 0.7

    # Network setup
    network = setup("Perceptron", num_in, num_out, None, layer_function, weights, alpha, binary_out)

    # Training
    mses, cverr, cycles = training(network, training_data, num_in, num_out)
    show_result(network, cycles, training_data, mses, cverr)

    # Computing
    compute(network, test_data, num_in, num_out)


def openai(env):
    if env == "Cartpole":
        # Data structure
        num_in = 4
        num_out = 1
        binary_out = True

        # Agent parameters
        layer_function = ["Step"]
        weights = [0.029999999999999583, 2.565418860536054, -1.7769954333529105, 9.968276281956078, -2.5429300705374094]
        alpha = 0.76

        # Network setup
        network = setup("Perceptron", num_in, num_out, None, layer_function, weights, alpha, binary_out)

        # Setup OpenAI Environment
        ai = OpenAI(network)

        # Training
        #        data, _ = getCSVData("data/cartpole_data.csv", share=1)
        #        training(network,data, num_in, num_out)
        #        show_result(network)

        # Runnign
        print("\n==============================================================")
        print("\n- OpenAI\n")
        ai.cartPole()


def main():
     arch = "MLP"
    
     if arch == "MLP":
         mlp()
     else:
         perceptron()

    # openai("Cartpole")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print("\n------\nError: {0}\n------".format(err))
        raise err
