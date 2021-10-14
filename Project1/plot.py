# -*- coding: utf-8 -*-
# IMPORTS
import matplotlib.pyplot as plt
import numpy as np

# SELF IMPORTS
from data import generator 
from lib.func import *
from lib.opt import *
from lib.utils import *
from lib.vis import *


load = np.load('data/output.npz')
print(load['arr_2'])

DX = [load['arr_1'], load['arr_1'], load['arr_2'], load['arr_3'], load['arr_4'], load['arr_5'], load['arr_6'], load['arr_7'], load['arr_8'], load['arr_9']]

colors = ['k','k--','r','r--','b','b--', 'g', 'g--', 'm', 'm--']
comparison(DX,colors)