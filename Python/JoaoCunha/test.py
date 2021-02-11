import mnist_loader
import network


# to measure exec time 
from timeit import default_timer as timer 


""" read the input data """
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

""" Initialize the network """
# test1
#net = network.Network([2, 3, 1])

# test2
#net = network.Network([784, 30, 10])

# test3
net = network.Network([784, 100, 10])

""" Train the network using Stochastic Gradient Descent """
# SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None)
#@numba.jit
net.SGD(training_data , 30, 10, 3.0, test_data=test_data)


