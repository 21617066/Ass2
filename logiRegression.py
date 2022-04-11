# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 09:59:47 2022

@author: EpEp
"""
import numpy as np
from scipy import optimize

def sig(x):
    return 1.0/(1.0 + np.exp(-x))
        
# x: dxN
# w: dx1
# y: 1xN 
class LogiRegression:
    def __init__(self):
        pass
            
    def grad_l(self ,w, X, y, N):
        # print("gw: " + str(w.shape))
        # print("gX: " + str(X.shape))
        # print("gy: " + str(y.shape))
        # print("gN: " + str(N))
        temp = sig(w.T @ X) - y
        temp2 = np.empty((2,N))
        # print("gt shape: " +  str(temp.shape))
        # print("gt2 shape: " +  str(temp2.shape))
        
        for n in range(N):
            # print("g" + str(n))
            temp2[:, n] = temp[:,n] * X[:,n]
        return np.sum(temp2, axis=1)

    def hessi_l(self, w, X, y, N):
        # print("hw: " + str(w.shape))
        # print("hX: " + str(X.shape))
        # print("hy: " + str(y.shape))
        # print("hN: " + str(N))
        temp = sig(w.T @ X) * (np.ones(N) - sig(w.T @ X))
        temp2 = np.empty((2,N))
        for n in range(N):
            # print("h" + str(n))
            temp2[:,n] = temp[:,n] * X[:,n]**2
        return np.sum(temp2, axis=1)
    
    # Function to find the root
    
    def newtonRaphson(self, w,X,y,N):
        h = self.grad_l(w,X,y,N) / self.hessi_l(w,X,y,N)
        i = 0
        while abs(h.all()) >= 0.0001:
            h = self.grad_l(w,X,y,N) / self.hessi_l(w,X,y,N)
            print(str(i) + "h: " + str(h))  
            i += 1
            # w(i+1) = w(i) - f(w) / f'(w)
            w = w - h
            print(str(i) + "w: " + str(w)) 
        
    def fit(self, X, y, N):
        w = np.array([0.5,0.5])
        w = w.reshape((2,1))
        print("w: " + str(w.shape))
        return self.newtonRaphson(w, X, y, N)
    
             
    

