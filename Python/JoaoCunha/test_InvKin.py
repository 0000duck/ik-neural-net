import InvKin_loader
import network
import pandas

# to measure exec time 
from timeit import default_timer as timer 


""" read the input data """
training_data, validation_data, test_data = InvKin_loader.load_data_wrapper()

""" Initialize the network """
# test1
net = network.Network([6, 1, 6])


""" Train the network using Stochastic Gradient Descent """
# SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None)
#@numba.jit
net.SGD(training_data , 30, 10, 3.0, test_data=0)
