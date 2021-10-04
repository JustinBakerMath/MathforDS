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

np.random.seed(0)

# DATA
A = generator.A
b = generator.b
coefs = generator.coefs
init = np.random.random(50)
lam = .001

#GRADIENT DESCENT
ittL2,nfeL2,dxL2 = desc(init, A, b, reg = l2Reg)
ittL1,nfeL1,dxL1 = desc(init, A, b, reg = LASSO)
#PROXIMAL GRADIENT
ittL3,nfeL3,dxL3 = desc(init, A, b, method = prox)
ittL4,nfeL4,dxL4 = desc(init, A, b, method = shrink)
#ACCEL PROX
ittL5,nfeL5,dxL5 = accel(init, A, b, method = accProx)
ittL6,nfeL6,dxL6 = accel(init, A, b, method = accShrink)

#PLOTTING
ITT = [ittL1, ittL2, ittL3, ittL4, ittL5, ittL6]
NFE = [nfeL1, nfeL2, nfeL3, nfeL4, nfeL5, nfeL6]
DX = [dxL1, dxL2, dxL3, dxL4, dxL5, dxL6]
colors = ['k','k--','r','r--','b','b--']
comparison(ITT,NFE,DX,6,colors)


"""
def F(x,lam = 0.001, order = 1, A=A,b=b):
    #given function to minimize
    F = 0
    for i in range(0,len(b)):
        F = F+1/2/len(b)*np.log(1+np.exp(-b[i]*np.dot(A[i],x)))
    F = F + lam*np.linalg.norm(x, ord=order)
    return F

def sgF(x,lam = 0.001, A=A,b=b):
    #subgradient of given function
    F = 0
    for i in range(0,len(b)):
        F = F-1/2/len(b)*np.exp(-b[i]*np.dot(A[i],x))*A[i]*b[i]\
            /(1+np.exp(-b[i]*np.dot(A[i],x)))
        F = F + lam*((x<0)*(-1)+(x>0)*(1)+(x==0)*0)
    return F

def optimal_step(x,sg,F,coefs = coefs):
    #function to return the optimal step size w/ known soln
    #inputs: 
        #x: current x value
        #coefs: known optimal solution
        #sg: subgradient
        #F: function
    #ouputs:
        #eta: optimal step size
    return (F(x)-F(coefs))/(np.linalg.norm(sg(x),ord=2))**2
    

def plot(y):
    #plots error in iterations over time
    #Inputs:
        #y: error over time
    #Ouputs:
        #plot of error over time
    x = np.array(range(1,len(y)+1))
    plt.plot(x, y)
    plt.xlabel("t")
    plt.ylabel("F(x^t)-F(x*)")
    plt.show()

def apply_sub_grad(initial,step_size=optimal_step,sg=sgF, coefs=coefs, F = F):
    #Sub-gradient implementation
    #Inputs
        #initial: initial vec (x1)
        #step_size: function for step size rule
        #sg: subgradient of function
        #coefs: actual solution
    #Ouputs
        #sol: solution
        #errs: F(xt)-F(x*)
    errs = []
    x = initial
    sol = initial
    i = 1
    while abs(F(x)-F(coefs))>1e-2:
        eta = step_size(x,sg,F)
        y = x - eta*sg(x)
        i+=1
        errs.append(F(y)-F(coefs))
        print(F(y)-F(coefs))
        if F(y)<F(x):
            sol = y
        x = y
        
    return errs,sol

def proxl1(x,lam=0.001):
    #proximal gradient operator for l1norm
    #inputs:
        #x: argument to evaluate at (vector)
        #lam: lambda parameter for soft-thresholding
    #ouputs:
        #\psi_st(x;\lam)
     return (x+lam)*(x<=-lam)+(x-lam)*(x>=lam)

def apply_prox_grad(initial, reg = proxl1, step_size = 1/2, grad = 0, f =0 ):
    #proximal gradient implementation for composite functions
    #inputs:
        #initial: initial vec
        #reg: regularization term for prox op
        #step_size: step size
        #grad: gradient of differentiable part
        #f: differentiable part of composite function
    #Ouputs
        #sol: solution
        #errs: F(xt)-F(x*)
    errs = []
    x = initial
    sol = initial
    i = 1
    while abs(F(x)-F(coefs))>1e-4:
        y = proxl1(x - step_size*grad(x))
        i+=1
        errs.append(F(y)-F(coefs))
        print(F(y)-F(coefs))
        if F(y)<F(x):
            sol = y
        x = y
        
    return errs,sol
######################Applying all methods 

initial = np.random.random(len(coefs))*max(abs(coefs))

(errs,sol)  = apply_sub_grad(initial, F = lambda x :F(x, lam=0.0001), sg= lambda x :sgF(x, lam=0.0001))
#(errs,sol)  = apply_sub_grad(initial)
plot(errs)

#(errsprox, solprox) = apply_prox_grad(initial)
#plot(errsprox)

x = np.random.random(50)
for i in range(10):
    dF = gradLR(x,A,b)+gradL2(x)
    x = gd(x,dF)
    print(x)

"""
