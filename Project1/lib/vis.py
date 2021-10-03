# -*- coding: utf-8 -*-
# IMPORTS
import matplotlib.pyplot as plt
import numpy as np

# MULTI PLOTS
def comparison(ITT,NFE,DX):
    """ Compares Iteration Accuracy and
        number of function evaluations """
    fig, ax = plt.subplots(1,2, tight_layout=True)

    for i in range(2):
        ax[0].plot(DX[i][1:])
        ax[1].scatter(ITT[i], NFE[i])

    ax[0].set_yscale('log')
    plt.show()

    return 1
