# -*- coding: utf-8 -*-
# IMPORTS
import matplotlib.pyplot as plt
import numpy as np

# MULTI PLOTS
def comparison(ITT,NFE,DX,n,colors=None):
    """ Compares Iteration Accuracy and
        number of function evaluations """
    fig, ax = plt.subplots(1,2, tight_layout=True)

    for i in range(n):
        ax[0].plot(DX[i][1:], colors[i])
        ax[1].scatter(ITT[i], NFE[i])

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    plt.show()

    return 1
