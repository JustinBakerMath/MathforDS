# -*- coding: utf-8 -*-
# IMPORTS
import matplotlib.pyplot as plt
import numpy as np

DPI = 80

# MULTI PLOTS
def comparison(X,DX,colors=None, labels = ["SubGrad L2",  "SubGrad L1", \
            "ProxGrad L2", "ProxGrad L1", \
            "Heavyball L2", "Heavyball L1", \
            "Nesterov L2", "Nesterov L1", \
            "ARNAG L2", "ARNAG L1"]):
    """ Compares Iteration Accuracy and
        number of function evaluations """
    n=len(DX)

    fig, ax = plt.subplots(1,1, tight_layout=True, dpi=DPI)

    for i in range(n):
        ax.plot(DX[i][1:], colors[i], label = labels[i])

    ax.set_yscale('log')
    fig.legend()
    plt.show()
    fig.savefig('./out/error.pdf')



    fig, ax = plt.subplots(1,1, tight_layout=True, dpi=DPI)
    for i,Xi in enumerate(X):
        norm = []
        for j in range(1,Xi.shape[0]):
            norm = norm+[np.linalg.norm(Xi[j]-Xi[j-1])]
        
        ax.plot(np.arange(len(norm)),norm, colors[i], label=labels[i])

    plt.show()
    fig.legend()
    fig.savefig('./out/dx.pdf')


    fig, ax = plt.subplots(1,1, tight_layout=True, dpi=DPI)
    for i,Xi in enumerate(X):
        norm = []
        for j in range(Xi.shape[0]):
            norm = norm+[np.linalg.norm(Xi[j],1)]
        
        ax.plot(np.arange(len(norm)),norm, colors[i], label=labels[i])

    plt.show()
    fig.legend()
    fig.savefig('./out/xl1.pdf')
    
    fig, ax = plt.subplots(1,1, tight_layout=True, dpi=DPI)
    for i,Xi in enumerate(X):
        norm = []
        for j in range(Xi.shape[0]):
            norm = norm+[np.linalg.norm(Xi[j],2)]
        
        ax.plot(np.arange(len(norm)),norm, colors[i], label=labels[i])

    plt.show()
    fig.legend()
    fig.savefig('./out/xl2.pdf')
    return 1
