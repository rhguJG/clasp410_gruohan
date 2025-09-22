#!/usr/bin/env python3

'''
This files solves the N-layer atmosphere problem for Lab 01 and all subparts.

TO REPRODUCE THE VALUES AND PLOPTS IN MY REPORT, DO THIS:
<tbd>
'''

import numpy as np
import matplotlib.pyplot as plt

# Physical Constants
sigma = 5.67E-8  #Units: W/m2/K-4

def n_layer_atmos(nlayers, epsilon=1, albedo=0.33, s0=1350, debug=False):
    '''
    Solve the N-layer atmosphere problem.

    Parameters:
        nlayers : int
            Number of atmospheric layers.
        epsilon : float
            Emissivity of each layer, defaulted to 1.
        albedo : float
            Planetary albedo, defaulted to 1.
        s0 : float
            Solar irradiance [W/m^2], defaulted to 1350.
        debug : bool
            If True, print matrices.
    
    Returns:
        temps : ndarray
            Temperatures [fluxes] of surface and layers.
    '''

    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # Populate based on our model:
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i == j:
                A[i, j] = -2 + 1 * (j == 0)
            else:
                A[i, j] = epsilon**(i>0) * (1-epsilon)**(np.abs(j - i) - 1)
    if (debug):
        print(A)
                    
    b[0] = -0.25 * s0 * (1-albedo)

    # Invert matrix:
    Ainv = np.linalg.inv(A) 
    # Get solution:
    fluxes = np.matmul(Ainv, b)

    temps = np.zeros_like(fluxes)
    temps[0] = np.power((fluxes[0] / sigma), 1/4)
    temps[1:] = np.power(fluxes[1:] / (sigma * epsilon), 1/4)
    return temps
