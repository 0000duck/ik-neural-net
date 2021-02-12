import numpy as np
import pandas as pd 
import pickle
import gzip


def load_data():
    ee_pos_ori = pd.read_csv(r'..\..\Dataset\UR10\ee_pos_ori.csv', sep = ';')
    joint_values = pd.read_csv(r'..\..\Dataset\UR10\joint_values.csv', sep = ';')

    dataset = np.array([ee_pos_ori, joint_values])

    training_data, validation_data, test_data = dataset[:5000, :5000], dataset[:3000, :3000], dataset[:1000, :1000]

    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()

    training_data = zip(tr_d[0], tr_d[1])
    validation_data = zip(va_d[0], va_d[1])
    test_data = zip(te_d[0], te_d[1])


    return (training_data, validation_data, test_data)

#training_data, validation_data, test_data = load_data()


