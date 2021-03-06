#Input
import pandas as pd 
import math
import numpy as np
ee_pos_ori = pd.read_csv(r'Dataset\UR10\ee_pos_ori.csv', sep = ';')
joint_values = pd.read_csv(r'Dataset\UR10\joint_values.csv', sep = ';')
# df_1 = ee_pos_ori.copy()
# df_2 = joint_values.copy()

def vectorize_it (df): 
    x = [np.reshape(x, (6, 1)) for x in df['x']] #Criar relação 1 para 1 entre entras e saídas em formato array
    y = [np.reshape(x, (6, 1)) for x in df['y']]
    final = zip(x, y) #Para replicar outro método
    return final

def list_it (df_1, df_2):
    df_1['join'] = df_1.index # Criar ligação entre os 2 ficheiro pela linha
    df_2['join'] = df_2.index
    
    df_new = df_2.merge(df_1, on = ['join'], how = 'inner') # Juntar os 2 data frames
    
    df_new['x'] = df_new.apply(lambda x: [x.xe, x.ye, x.ze, x.alpha, x.beta, x.gamma], axis = 1) # Juntar as 6 variaveis numas lista
    df_new['y'] = df_new.apply(lambda x: [x.theta1, x.theta2, x.theta3, x.theta4, x.theta5, x.theta6], axis = 1)
    
    size = len(df_new) # Tamanho total para a seguir dividir
    
    train = df_new.iloc[0:math.ceil(0.7 * size)]
    validation = df_new.iloc[math.ceil(0.7 * size) : math.ceil(0.9 * size)]
    test = df_new.iloc[math.ceil(0.9 * size) : size]

    train_data = vectorize_it(train) #Função acima
    validation_data = vectorize_it(validation)
    test_data = vectorize_it(test)
    return train_data, validation_data, test_data

training_data, validation_data, test_data = list_it(ee_pos_ori, joint_values)

# ---------------------
# - network.py example:
import sys
sys.path.insert(0, './Python/JoaoCunha') # Para mudar para a pasta onde estão os ficheiros - Estava a trabalhar na pasta ik-neural-net
import network

net = network.Network([6, 30, 6])
net.SGD(training_data, 100, 10, 3.0, test_data=test_data)

#New to inverse cinematic
net = network.Network([784, 30, 10])
net.SGD(training_data, 100, 10, 3.0, test_data=test_data)



# ----------------------
# - network2.py example:
import network2

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)

# chapter 3 - Overfitting example - too many epochs of learning applied on small (1k samples) amount od data.
# Overfitting is treating noise as a signal.
'''
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True)
'''

# chapter 3 - Regularization (weight decay) example 1 (only 1000 of training data and 30 hidden neurons)
'''
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data[:1000], 400, 10, 0.5,
    evaluation_data=test_data,
    lmbda = 0.1, # this is a regularization parameter
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True)
'''

# chapter 3 - Early stopping implemented
'''
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data[:1000], 30, 10, 0.5,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    early_stopping_n=10)
'''

# chapter 4 - The vanishing gradient problem - deep networks are hard to train with simple SGD algorithm
# this network learns much slower than a shallow one.
'''
net = network2.Network([784, 30, 30, 30, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.1,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)
'''


# ----------------------
# Theano and CUDA
# ----------------------

"""
    This deep network uses Theano with GPU acceleration support.
    I am using Ubuntu 16.04 with CUDA 7.5.
    Tutorial:
    http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu

    The following command will update only Theano:
        sudo pip install --upgrade --no-deps theano

    The following command will update Theano and Numpy/Scipy (warning bellow):
        sudo pip install --upgrade theano

"""

"""
    Below, there is a testing function to check whether your computations have been made on CPU or GPU.
    If the result is 'Used the cpu' and you want to have it in gpu,     do the following:
    1) install theano:
        sudo python3.5 -m pip install Theano
    2) download and install the latest cuda:
        https://developer.nvidia.com/cuda-downloads
        I had some issues with that, so I followed this idea (better option is to download the 1,1GB package as .run file):
        http://askubuntu.com/questions/760242/how-can-i-force-16-04-to-add-a-repository-even-if-it-isnt-considered-secure-eno
        You may also want to grab the proper NVidia driver, choose it form there:
        System Settings > Software & Updates > Additional Drivers.
    3) should work, run it with:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python3.5 test.py
        http://deeplearning.net/software/theano/tutorial/using_gpu.html
    4) Optionally, you can add cuDNN support from:
        https://developer.nvidia.com/cudnn


"""
def testTheano():
    from theano import function, config, shared, sandbox
    import theano.tensor as T
    import numpy
    import time
    print("Testing Theano library...")
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], T.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')
# Perform check:
#testTheano()


# ----------------------
# - network3.py example:
import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer # softmax plus log-likelihood cost is more common in modern image classification networks.

# read data:
training_data, validation_data, test_data = network3.load_data_shared()
# mini-batch size:
mini_batch_size = 10

# chapter 6 - shallow architecture using just a single hidden layer, containing 100 hidden neurons.
'''
net = Network([
    FullyConnectedLayer(n_in=784, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
'''

# chapter 6 - 5x5 local receptive fields, 20 feature maps, max-pooling layer 2x2
'''
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2)),
    FullyConnectedLayer(n_in=20*12*12, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
'''

# chapter 6 - inserting a second convolutional-pooling layer to the previous example => better accuracy
'''
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2)),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2)),
    FullyConnectedLayer(n_in=40*4*4, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
'''

# chapter 6 -  rectified linear units and some l2 regularization (lmbda=0.1) => even better accuracy
from network3 import ReLU
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)
