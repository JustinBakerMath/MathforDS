# -*- coding: utf-8 -*-
# IMPORTS
import matplotlib.pyplot as plt
import numpy as np

DPI = 80

# MULTI PLOTS
def comparison(DX,colors=None):
    """ Compares Iteration Accuracy and
        number of function evaluations """
    n=len(DX)
    labels = ["SubGrad L2",  "SubGrad L1", \
            "ProxGrad L2", "ProxGrad L1", \
            "Heavyball L2", "Heavyball L1", \
            "Nesterov L2", "Nesterov L1", \
            "ARNAG L2", "ARNAG L1"]
    fig, ax = plt.subplots(1,1, tight_layout=True, dpi=DPI)

    for i in range(n):
        ax.plot(DX[i][1:], colors[i], label = labels[i])

    ax.set_yscale('log')
    fig.legend()
    plt.show()
    fig.savefig('./out/error.pdf')

    return 1
