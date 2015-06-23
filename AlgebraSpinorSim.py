# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:45:55 2015
Quantum package to evaluate symbolic raising and lowering operators
@author: zag
"""
from sympy import Symbol, sqrt
import copy

class AngularKet(object):
    """object which contains a coefficient and values of L and M"""
    def __init__(self,L,M):
        self.L = L
        self.M = M
        self.coef = 1
        
    def lowerM(self):
        if self.M <1:
            self.coef = 0
        else:
            self.M = self.M - 1
            self.coef = self.coef * sqrt(self.L*(self.L+1)-self.M*(self.M-1))
        
    def raiseM(self):
        if self.M > self.L:
            self.coef = 0
        else:
            self.M = self.M+- 1
            self.coef = self.coef * sqrt(self.L*(self.L+1)-self.M*(self.M+1))
        
    def pretty_print(self):
        print('{0}|{1},{2}>'.format(self.coef,self.L,self.M))
        
        
class FockKet(object):
    """Fock ket"""
    def __init__(self,Nm,N0,Np):
        self.state = {'-1' : Nm,'0' : N0,'1':Np}
        self.coef = 1
        
    def destoryN(self, p):
        if p not in self.state.keys():
            print('Not a correct p')
        else:
            self.coef = self.coef * sqrt(self.state[p])
            self.state[p] = self.state[p] -1
            
    def createN(self, p):
        if p not in self.state.keys():
            print('Not a correct p')
        else:
            self.coef = self.coef * sqrt(sqrt(self.state[p])+1)
            self.state[p] = self.state[p] + 1
            
    def pretty_print(self):
        print('{0}|{1},{2},{3}>'.format(self.coef,self.state['-1'],self.state['0'],self.state['1']))


def raising_Operator(ket):
    """raising operator to act on kets"""
    if type(ket) == FockKet:
        pass
    elif type(ket) == AngularKet:
        ket.raiseM()

if __name__ == '__main__':
    ket = FockKet(1,2,1)
    
    ket.pretty_print()
    ket.createN('1')
    ket.pretty_print()
    
    
