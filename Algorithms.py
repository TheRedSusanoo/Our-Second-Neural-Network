import numpy as np
import random


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def derivative_of_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork(object):

    def __init__(self, neuron_counts): # __init__(self) is kind of analogous to a parameterized constructor in C++, except member variables are declared(and initialized) IN this function
        self.layer_count = len(neuron_counts)   # Number of layers
        self.biases = [np.random.randn(y, 1) for y in neuron_counts[1:]]    #Assign a random bias to each neuron in every layer
        self.weights = [np.random.randn(y, x) for x, y in zip(neuron_counts[:-1], neuron_counts[1:])]   #Assign random weights to each connection in the network
        self.neuron_counts = neuron_counts  # Number of neurons in each layer of the network


    def feed_forward(self, each_training_image, training): #takes an image and propagates it forward to the output layer by calculating activations along the way
        current_activations = each_training_image # Pixel values of a 28x28 image input at the first layer of the network
        activations = [current_activations] # List to store all the activations at each training OR testing image
        z_values = []   # list to store the 'z' values for each training image

        for each_bias, each_weight in zip(self.biases, self.weights):
            z = np.dot(each_weight, current_activations) + each_bias    # z = wa+b
            z_values.append(z)
            current_activations = sigmoid(z)
            activations.append(current_activations)
        if training == 1:       # If the function is called during training return all the activations and z values in the network so they can be used for backprop
            return activations, z_values
        else:           # else if the function is called during testing, return only the output activations
            return activations[-1]



    def backpropagate_the_error(self, activations, z_values, expected_output):

        dabba_C_by_dabba_b = [np.zeros(b.shape) for b in self.biases]
        dabba_C_by_dabba_w = [np.zeros(w.shape) for w in self.weights]
        """Make two containers having the same shape as that of weights and biases to later store the partial
        derivatives of all the weights and biases in the network at the same correspondent indices as the actual weights and biases
        so it becomes easier to tweak them"""


        """Following code implements the four fundamental equations of backpropagation"""

        error_delta = (activations[-1] - expected_output) * derivative_of_sigmoid(z_values[-1])
        """This is Equation 1, which is used to calculate the error at the output layer using derivative of the cost function
        which is just the difference between actual and desired output"""

        dabba_C_by_dabba_b[-1] = error_delta
        """This is Equation 3, which states that the partial derivative of the cost function wrt the bias at a 
        particular neuron in the lth layer is equal to the value delta"""

        dabba_C_by_dabba_w[-1] = np.dot(error_delta, activations[-2].transpose())
        """This is Equation 4, which states that the partial derivative of the cost function wrt every weight at a 
        particular neuron in the lth layer is equal to the value delta multiplied by the transpose of the matrix of 
        activations of the previous(l-1) layer"""


        """NOTE: Here we have only calculated the partial derivatives wrt bias and weights at the output layer,
        Derivatives for the previous layers will be calculated in the following code inside a loop using t
        he recurrence relation of equation 2"""


        for every_layer in range(2, self.layer_count):
            z = z_values[-every_layer] # from the last layer to the 2nd layer
            derivative_of_z = derivative_of_sigmoid(z)
            error_delta = np.dot(self.weights[-every_layer+1].transpose(), error_delta) * derivative_of_z
            """This is Equation 3, which lets us compute the error or delta(S) for each layer in terms of the error 
            in the next layer. The value of the error in turn then lets us compute the partial derivatives dC/db and 
            dC/dw for each neuron in each of the respective layers"""



            dabba_C_by_dabba_b[-every_layer] = error_delta # Equation 3
            dabba_C_by_dabba_w[-every_layer] = np.dot(error_delta, activations[-every_layer-1].transpose()) #Equation 4

        return dabba_C_by_dabba_b, dabba_C_by_dabba_w # Return all the partial derivatives for all neurons in the network




    def stochastic_gradient_descent(self, training_data, learning_rate, epochs, batch_size, test_data=None):

        test_count = len(test_data)
        n = len(training_data)

        for i in range(epochs): # For each epoch
            random.shuffle(training_data) # Shuffle the training data
            batches = [training_data[j:j+batch_size] for j in range(0, n, batch_size)] # Split the training data in mini batches of size 'batch_size'
            for every_batch in batches:
                dabba_C_by_dabba_b = [np.zeros(b.shape) for b in self.biases]
                dabba_C_by_dabba_w = [np.zeros(w.shape) for w in self.weights]
                """Make containers initialized to zeros having the same dimensions as 'weights' and 'biases' to later store the 
                partial derivatives dC/db and dC/dw"""

                for x, y in every_batch:
                    activations, z_values = self.feed_forward(x, 1)
                    #feed a training image into the network, which will return all the activations and 'z's
                    new_dabba_C_by_dabba_b, new_dabba_C_by_dabba_w = self.backpropagate_the_error(activations, z_values, y)
                    #backpropagate_the_error returns all the partial derivatives for each neuron for current training sample

                    dabba_C_by_dabba_b = [initial_values_b + fresh_derivatives_b for initial_values_b, fresh_derivatives_b in zip(dabba_C_by_dabba_b, new_dabba_C_by_dabba_b)]
                    dabba_C_by_dabba_w = [initial_values_w + fresh_derivatives_w for initial_values_w, fresh_derivatives_w in zip(dabba_C_by_dabba_w, new_dabba_C_by_dabba_w)]

                self.biases = [every_bias - (learning_rate / len(every_batch)) * dabba_b for every_bias, dabba_b in zip(self.biases, dabba_C_by_dabba_b)]
                self.weights = [every_weight - (learning_rate / len(every_batch)) * dabba_w for every_weight, dabba_w in zip(self.weights, dabba_C_by_dabba_w)]

                """Tweak the biases and weights in the network by using the partial derivatives, learning rate and mini batch size"""

            print ("Epoch {0}: {1} images classified correctly out of {2} images".format(i+1, self.test_network(test_data), len(test_data)))


    def test_network(self, test_data):
        test_results = [(np.argmax(self.feed_forward(every_image, 0)),expected_output) for (every_image, expected_output) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


