# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:39:22 2015
Plotter
@author: zag
"""
import numpy as np
import matplotlib.pyplot as plt

#filename = 'results.txt'
filename = 'results_multi.txt'

with open(filename,'r') as fp:
    data_in = fp.readlines()
    
    
ind = data_in.index('{:<15}{:<15}{:<15}{:<15}\n'.format('t(s)','mean','stddev','norm'))

data = np.zeros((len(data_in[ind+1:]),4))
for i,j in enumerate(data_in[ind+1:]):
    data[i] = np.asarray([float(i) for i in j.rstrip('\n').split()])

fig, ax = plt.subplots(1,1)
ax.errorbar(data[:,0],data[:,1],yerr = data[:,2])
ax.set_xlabel('Time (s)')
ax.set_ylabel('N in m=0')
ax.set_title('Spinor Reversal: wavefunction recovered is {:>5.2f}'.format(np.mean(data[:,3])))
plt.show()
