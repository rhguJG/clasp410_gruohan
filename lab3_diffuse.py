#!/usr/bin/env python3

'''
Tools and methods for completing Lab 3 which is the best lab.
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


def solve_heat(xstop=1, tstop=0.2, dx=0.2, dt=0.02, c2=1, bc='dirichlet'):
    '''
    A function for solving the heat equation

    Parameters
    ----------
    Fill this out don't forget. :P
    c2 : float
        c^2, the square of the diffusion coefficient.

    Returns
    -------
    x, t : 1D Numpy arrays
        Space and time values, respectively.
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime
    '''
    

    # Get grid sizes:
    N = int(tstop / dt)
    M = int(xstop / dx)

    # Set up space and time grid:
    t = np.linspace(0, tstop, N)
    x = np.linspace(0, xstop, M)

    # Create solution matrix; set initial conditions
    U = np.zeros([M, N])
    U[:, 0] = 4*x - 4*x**2

    # Get our "r" coeff:
    r = c2 * (dt/dx**2)

    # Solve our equation!
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])

        # Apply boundary conditions at the new time level
        if bc == 'dirichlet':
            U[0, j+1] = 0.0
            U[M-1, j+1] = 0.0
        elif bc == 'neumann':
            U[0, j+1] = U[1, j+1]
            U[M-1, j+1] = U[M-2, j+1]

    # Return our pretty solution to the caller:
    return t, x, U


def plot_heatsolve(t, x, U, title=None, **kwargs):
    '''
    Plot the 2D solution for the `solve_heat` function.

    Extra kwargs handed to pcolor.

    Paramters
    ---------
    t, x : 1D Numpy arrays
        Space and time values, respectively.
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime
    title : str, default is None
        Set title of figure.

    Returns
    -------
    fig, ax : Matplotlib figure & axes objects
        The figure and axes of the plot.

    cbar : Matplotlib color bar object
        The color bar on the final plot
    '''

    # Check our kwargs for defaults:
    # Set default cmap to hot
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'hot'

    # Create and configure figure & axes:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Add contour to our axes:
    contour = ax.pcolor(t, x, U, **kwargs)
    cbar = plt.colorbar(contour)

    # Add labels to stuff!
    cbar.set_label(r'Temperature ($^{\circ}C$)')
    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Position ($m$)')
    ax.set_title(title)

    fig.tight_layout()

    return fig, ax, cbar