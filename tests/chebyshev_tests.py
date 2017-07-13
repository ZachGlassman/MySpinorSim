import numpy as np
import SpinorBECSimulation.CoherentStateChebyshev.chebyshev_functions as cf
import SpinorBECSimulation.CoherentStateChebyshev.chebyshev_functions_numba as cf_n

from nose.tools import assert_equal

complex_arrays = [
            (np.array([1+2j, 3+2j]), 2),
            (np.array([1, 10]), 4)
        ]

class test_moments(object):
    outputs = [
        (62.0, 228.0),
        (604, 3616)
        ]
    
    def test_numba(self):
        for arr, out in zip(complex_arrays, self.outputs):
            assert_equal(cf_n.moments(*arr), out)
            
    def test_no_numba(self):
        for arr, out in zip(complex_arrays, self.outputs):
            assert_equal(cf.moments(*arr), out)
        
    