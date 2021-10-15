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

init = np.random.uniform(-2,2,50)
itrs = 1000
lam = 0.001
eta = 1e-5

lopt = generator.coefs
f2 = linReg(lopt,A,b)+l2Reg(lopt,lam)
f1 = linReg(lopt,A,b)+l1Reg(lopt,lam)

X9, dxL9 = accel(init, A, b, f2, lam=lam, eta=eta, method=nestProx, reg=l2Reg, gradReg=gradl2, restart=True, nesterov=True, itrs=itrs*5)
X10, dxL10 = accel(init, A, b, f1, lam=lam, eta=eta, method=nestProx, reg=l1Reg, gradReg=gradl1, restart=True, nesterov=True, itrs=itrs*5)
f2 = linReg(X9[-1],A,b)+l2Reg(X9[-1],lam)
f1 = linReg(X10[-1],A,b)+l1Reg(X10[-1],lam)

print("Generated Optimal Values")
#SUB-GRADIENT
print("Subgradient L2")
X1, dxL1 = desc(init, A, b, f1, lam=lam, eta=eta, reg=l2Reg, gradReg=gradl2, itrs=itrs)
print("Subgradient L1")
X2, dxL2 = desc(init, A, b, f2, lam=lam, eta=eta, reg=l1Reg, gradReg=gradl1, itrs=itrs)
#PROX-GRADIENT
print("Proximal gradient L2")
X3, dxL3 = desc(init, A, b, f1, lam=lam, eta=eta, method=prox, reg=l2Reg, itrs=itrs)
print("Proximal gradient L1")
X4, dxL4 = desc(init, A, b, f2, lam=lam, eta=eta, method=shrink, reg=l1Reg, itrs=itrs)
#HBALL-PROX
print("Heavyball proximal gradient L2")
X5, dxL5 = accel(init, A, b, f2, lam=lam, eta=eta, method=hbProx, reg=l2Reg, itrs=itrs)
print("Heavyball proximal gradient L1")
X6, dxL6 = accel(init, A, b, f1, lam=lam, eta=eta, method=hbShrink, reg=l1Reg, itrs=itrs)
#NEST-PROX
print("Nesterov proximal gradient L2")
X7, dxL7 = accel(init, A, b, f2, lam=lam, eta=eta, method=nestProx, reg=l2Reg, gradReg=gradl2, nesterov=True, itrs=itrs)
print("Nesterov proximal gradient L1")
X8, dxL8 = accel(init, A, b, f1, lam=lam, eta=eta, method=nestProx, reg=l1Reg, gradReg=gradl1, nesterov=True, itrs=itrs)
#RESTART-NEST
print("Adaptive restart Nesterov proximal gradient L2")
X9, dxL9 = accel(init, A, b, f2, lam=lam, eta=eta, method=nestProx, reg=l2Reg, gradReg=gradl2, restart=True, nesterov=True, itrs=itrs)
print("Adaptive restart Nesterov proximal gradient L1")
X10, dxL10 = accel(init, A, b, f1, lam=lam, eta=eta, method=nestProx, reg=l1Reg, gradReg=gradl1, restart=True, nesterov=True, itrs=itrs)

#PLOTTING
X = [X1, X2, X3, X4, X5, X6, X7, X8, X9, X10]
DX = [dxL1, dxL2, dxL3, dxL4, dxL5, dxL6, dxL7, dxL8, dxL9, dxL10]
np.savez('data/x', X1, X2, X3, X4, X5, X6, X7, X8, X9, X10)
np.savez('data/err', dxL1, dxL2, dxL3, dxL4, dxL5, dxL6, dxL7, dxL8, dxL9, dxL10)
colors = ['k','k--','r','r--','b','b--', 'g', 'g--', 'm', 'm--']
comparison(X,DX,colors)