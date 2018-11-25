import numpy as np
import prog
import read_minst_data
import  random

#delete something ?

training_data, validation_data, test_data = read_minst_data.prep_data()
print("Finished loading and preparing the data\n")
my_neural_net = prog.NeuralNetwork([784,30,10])
print("Initialized the Neural Network\n")
my_neural_net.stochastic_gradient_descent(list(training_data),3.0, 30, 10, list(test_data) )