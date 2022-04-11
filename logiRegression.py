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
            
    def grad_l(self ,w, X, y, N, lam):
        sigmod = sig(w.T @ X[:,0])
        print((1/lam)*w)
        temp = (sigmod - y[0]) * X[:,0] + (1/lam)*w.T
        
        for n in range(N-1):
            sigmod = sig(w.T @ X[:,n+1])
            temp = temp + (sigmod - y[n+1]) * X[:,n+1] + (1/lam)*w.T
        
        #return np.sum(temp2, axis=1)
        print(temp)
        return temp

    def hessi_l(self, w, X, N, lam):
        sigmod = sig(w.T @ X[:,0])
        print((1/lam)*np.ones(w.shape))
        temp = sigmod * (1 - sigmod)*X[:,0]**2 + (1/lam)*np.ones(w.T.shape)
        
        for n in range(N-1):
            # print("h" + str(n))
            sigmod = sig(w.T @ X[:,n+1])
            temp = temp + sigmod * (1 - sigmod)*X[:,n+1]**2 + (1/lam)*np.ones(w.T.shape)
        
        #return np.sum(temp2, axis=1)
        print(temp)
        return temp
    
    # Function to find the root
    
    def newtonRaphson(self, w,X,y,N,lam):
        h = self.grad_l(w,X,y,N,lam) / self.hessi_l(w,X,N,lam)
        h = h.reshape(w.shape)
        i = 0
        while abs(h[0,0]) >= 0.001 or abs(h[1,0]) >= 0.001:
            h = self.grad_l(w,X,y,N,lam) / self.hessi_l(w,X,N,lam)
            h = h.reshape(w.shape)
            # w(i+1) = w(i) - grad_l(w) / hessi_l(w)
            w = w - h
            print(str(i) + "w: " + str(w.T))
            i += 1
        return w
        
    def fit(self, w, X, y, N, lam):
        print("w: " + str(w.T.shape))
        print("X: " + str(X.shape))
        print("y: " + str(y.shape))
        self.w = self.newtonRaphson(w.T, X, y, N, lam)
        return self.w
    
    def predict(self, X):
        pc_one_x = sig(self.w.T @ X)
        self.prediction = np.round(pc_one_x).astype(int)
    
             
    

