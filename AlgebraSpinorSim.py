# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:45:55 2015
Quantum package to evaluate symbolic raising and lowering operators
@author: zag
"""
#from sympy import  sqrt
import sympy
import sympy.mpmath as mpmath
import time
from numpy import sqrt 
import sys

class AngularKet(object):
    """object which contains a coefficient and values of L and M"""
    def __init__(self,L,M, coef = 1):
        self.L = L
        self.M = M
        self.coef = mpmath.mpf(coef)
        
    def lowerM(self):
        if self.M <1:
            return 0
        else:
            M = self.M - 1
            coef = self.coef * sqrt(self.L*(self.L+1)-self.M*(self.M-1))
            return AngularKet(self.L,M,coef)
        
    def raiseM(self):
        if self.M > self.L:
            return 0
        else:
            M = self.M + 1
            coef = self.coef * sqrt(self.L*(self.L+1)-self.M*(self.M+1))
            return AngularKet(self.L,M,coef)
        
    def pretty_print(self):
        print('{0}|{1},{2}>'.format(self.coef,self.L,self.M))
        
    def __str__(self):
        return '{0}|{1},{2}>'.format(self.coef,self.L,self.M)
        
        
class FockKet(object):
    """Fock ket"""
    def __init__(self,coef,Nm,N0,Np):
        self.state = {'Nm' : Nm,'N0' : N0,'Np':Np}
        self.coef = mpmath.mpf(coef)
        
    def destroyN(self, p):
        if p not in self.state.keys():
            print('Not a correct p')
        else:
            state = self.state.copy()
            c = self.coef * sqrt(state[p])
            state[p] = state[p] - 1
            return FockKet(c,**state)
            
    def createN(self, p):
        if p not in self.state.keys():
            print('Not a correct p')
        else:
            state = self.state.copy()
            c = self.coef * sqrt(state[p]+1)
            state[p] = state[p] + 1
            return FockKet(c,**state)
            
    def pretty_print(self):
        print(self.__str__())
        
    def __str__(self):
        return '{0}|{1},{2},{3}>'.format(self.coef,self.state['Nm'],self.state['N0'],self.state['Np'])
        
    def __add__(self, other):
        """at most need to add two states of same states"""
        if self.coef == 0:
            return other
        elif other.coef == 0:
            return self
        else:
            c = self.coef * other.coef
            return FockKet(c, **self.state.copy())
        
    def __mul__(self,other):
        """multiply by scaler"""
        return FockKet(self.coef * other, **self.state)
    
    def __rmul__(self,other):
        """multiply by scaler on right"""
        return FockKet(self.coef * other, **self.state)
        

class SuperState(object):
    """superposition of fock kets"""
    def __init__(self, ket = None):
        if isinstance(ket,AngularKet) or isinstance(ket,FockKet):
            self.kets = [ket]
        else:
            self.kets = [] 
        
    def add_state(self,ket):
        """add a ket to the State"""
        statin = False
        for i in self.kets:
            if ket.state == i.state:
                i = i + ket
                statin = True
                break
        if not statin and ket.coef != 0:
            self.kets.append(ket)
        
    def multiply(self,val):
        temp = [i * val for i in self.kets]
        self.kets = temp         
        
    def raising_Operator(self,ket):
        """raising operator to act on kets, returns states"""
        if type(ket) == FockKet:
            self.add_state(sqrt(2)*ket.destroyN('N0').createN('Np'))
            self.add_state(sqrt(2)*ket.destroyN('Nm').createN('N0'))
        elif type(ket) == AngularKet:
            self.add_state(ket.raiseM())
            
    def raise_state(self):
        temp = [i for i in self.kets]
        self.kets = []
        for i in temp:
            self.raising_Operator(i)
    
    def get_max(self):
        return max([abs(i.coef) for i in self.kets])
       
    def __str__(self):
        return ''.join(i.__str__() + '+' for i in self.kets)
        
    def pretty_print(self):
        print([i.__str__() for i in self.kets])
    
    def filter_low(self,epsilon):
        temp = [i for i in self.kets if abs(i.coef) > epsilon]
        self.kets = temp
        
#fancy writeout
def write_progress(step,total):
    #write out fancy
    perc_done = step/(total) * 100
    #50 character string always
    num_marks = int(.5 * perc_done)
    out = ''.join('#' for i in range(num_marks))
    out = out + ''.join(' ' for i in range(50 - num_marks))
    sys.stdout.write('\r[{0}]{1:>2.0f}%'.format(out,perc_done))
    sys.stdout.flush()
    
    
if __name__ == '__main__':
    #start with the stretched state and apply the L+ operator L times
    N = 40000

   
    with open('StateOut.txt','w') as fp:

        fock = SuperState(FockKet(1,N,0,0)) 
        ang = AngularKet(N,-N)
        s = time.time()
        fp.write('{0} : {1}\n'.format(ang.__str__(),fock.__str__()))
        for i in range(N):  
            write_progress(i+1,N)
            fock.raise_state()
            ang = ang.raiseM()
            fock.multiply(1/ang.coef)
            ang.coef = 1
            #norm = fock.get_max()
            #fock.multiply(1/norm)
            norm = len([i.coef for i in fock.kets if abs(i.coef) > 1e-200])
            fock.filter_low(1e-200)
            fp.write('{2},{3} : {0} = {1}\n'.format(ang.__str__(),fock.__str__(),norm,len(fock.kets)))
            
        e = time.time()
    print('\n',e-s)
   
    
    
    
