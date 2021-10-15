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
xload = np.load('data/x.npz')
X = [xload['arr_0'], xload['arr_1'], xload['arr_2'], xload['arr_3'], xload['arr_4'], xload['arr_5'], xload['arr_6'], xload['arr_7'], xload['arr_8'], xload['arr_9']]
DX = [load['arr_0'], load['arr_1'], load['arr_2'], load['arr_3'], load['arr_4'], load['arr_5'], load['arr_6'], load['arr_7'], load['arr_8'], load['arr_9']]
labels = ["SubGrad L2",  "SubGrad L1", \
            "ProxGrad L2", "ProxGrad L1", \
            "Heavyball L2", "Heavyball L1", \
            "Nesterov L2", "Nesterov L1", \
            "ARNAG L2", "ARNAG L1"]
colors = ['k','k--','r','r--','b','b--', 'g', 'g--', 'm', 'm--']
comparison(X,DX,colors, labels)