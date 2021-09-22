# -*- coding: utf-8 -*-

#IMPORTS
from lib.utils import *

# LOAD DATA
A,b = load_data('./data/data.npz')

# OUTPUT
print('Recieved A data of shape ', A.shape)
print('Recieved b data of shape ', b.shape)
print('Input Data (concatenated)\n', A[:10,:3],'\nLabels\n', b[:10])