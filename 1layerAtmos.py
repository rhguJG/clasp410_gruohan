#!/usr/bin/env python3
'''
Test the hypothesis that global warming is fully explainable by an increase in solar forcing.
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Define global constants:
sigma = 5.67E-8

# Values we will need in our solar forcing exploration problem.
year = np.array([1900, 1950, 2000])
s0 = np.array([1365, 1366.5, 1368]) # Solar forcing in W/m2
t_anom = np.array([-.4, 0, .4])     # Temperature anomaly since 1950 in C

def temp_1layer(s0=1365.0, albedo=0.33, epsilon=1):
    '''
    Given solar forcing (s0) and albedo, determine the temperature of the Earth's
    surface using a single-layer perfectly absorbing energy balanced
    atmosphere model.

    Parameters
    ----------
    albedo : float, default=0.33
        Set the surface albedo to reflect incoming solar shortwave radiation.
    s0 : float, default=1350
        Incoming solar irradiance in W/m^2.
    epsilon : float, default=1.0
        Set emission of single layer atmosphere.

    Returns
    -------
    te : float
        Surface temperature in Kelvin
    '''

    te = (s0* (1-albedo) / (2*sigma))**(1/4.)

    return te

def compare_warming():
    '''
    Create a figure to test if changes in solar driving can account for 
    climate change.
    '''

    t_model = temp_1layer(s0 = s0)
    t_obs = t_model[1] + t_anom

    fig, ax = plt.subplots(1, 1, figsize=(8,8))

    ax.plot(year, t_obs, label="Observed Temperature Change")
    ax.plot(year, t_model, label="Predicted Temperature Change")

    ax.legend(loc='best')
    ax.set_xlabel('Year')
    ax.set_ylabel('Surface Temperature ($K$)')
    ax.set_title('Do Solar Force affect global warming?')

    fig.tight_layout()