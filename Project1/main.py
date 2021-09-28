# -*- coding: utf-8 -*-

import numpy as np
from data import generator 
import matplotlib.pyplot as plt

A = generator.A
b = generator.b
coefs = generator.coefs

initial = np.random.random(len(coefs))*max(abs(coefs))


def F(x,lam = 0.001, order = 1, A=A,b=b):
    #given function to minimize
    F = 0
    for i in range(0,len(b)):
        F = F+1/2/len(b)*np.log(1+np.exp(-b[i]*np.dot(A[i],x)))
    F = F + np.linalg.norm(x, ord=order)
    return F

def sgF(x,A=A,b=b):
    #subgradient of given function
    F = 0
    for i in range(0,len(b)):
        F = F-1/2/len(b)*np.exp(-b[i]*np.dot(A[i],x))*A[i]*b[i]\
            /(1+np.exp(-b[i]*np.dot(A[i],x)))
        F = F + (x<0)*(-1)+(x>0)*(1)+(x==0)*0
    return F

def optimal_step(t,x,coefs = coefs):
    #function to return the optimal step size w/ known soln
    #inputs: 
        #t: current step
        #x: current x value
        #coefs: known optimal solution
    #ouputs:
        #eta: optimal step size
    return (F(x)-F(coefs))/(np.linalg.norm(sgF(x))**2)
    

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

def apply_sub_grad(initial,step_size=optimal_step,sg=sgF, coefs=coefs):
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
        eta = step_size(i,x)
        y = x - eta*sg(x)
        i+=1
        errs.append(F(y)-F(coefs))
        print(F(y)-F(coefs))
        print(F(y)-F(coefs))
        if F(y)<F(x):
            sol = y
        x = y
        
    return errs,sol

(errs,sol)  = apply_sub_grad(initial)
plot(errs)
