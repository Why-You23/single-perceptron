# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:26:15 2022

@author: UMAR SADIQUE
"""

import numpy as np
"""Implements a perceptron network"""
class Perceptron(object): # create class for object
    def __init__(self, input_size, lr=0.1, epochs=7, bias = 0): # initialize input for class (self is for object for which we create class)
        #self.W = np.zeros(input_size+1) # randomly initilize weights
        self.W = np.array([0, 0, bias]) # initialize weights and bias value
        self.epochs = epochs #
        self.lr = lr #
        self.bias = bias
        
    def activation_fn(self, x):
        #return (x >= 0).astype(np.float32)
        return 1 if x >= 0.5 else 0
    
    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
    
    def fit(self, X, d):
        for _ in range(self.epochs):
            #print("weiths after ", _+1 ," epoch", self.W)
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                
                self.W = self.W + self.lr * e * x
                self.W[2] = self.bias
                
                
" NAND gate optimization"    
if __name__ == '__main__':
    print("\n\ncalculation for NAND gate")
    X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
    d = np.array([1, 1, 1, 0])
    perceptron = Perceptron(input_size=2, lr= 0.1, bias = 1)
    perceptron.fit(X, d)
    print("Optimize weights for NAND gate", perceptron.W)