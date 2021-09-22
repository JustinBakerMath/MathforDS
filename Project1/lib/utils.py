# -*- coding: utf-8 -*-
import numpy as np
import sys

sys.path.append('../')

def load_data(fname):

    with open(fname, 'rb') as f:
        data = np.load(f)['arr_0']
        
    A,b = data[:-1,:], data[-1,:]
    
    return A.T,b


if __name__ == "__main__":

    A,b = load_data('./data/data.npz')

    print('Recieved A data of shape ', A.shape)
    print('Recieved b data of shape ', b.shape)

    print('Input Data (concatenated)\n', A[:10,:3],'\nLabels\n', b[:10])
