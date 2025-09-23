#!/usr/bin/env python3

'''
This files solves the N-layer atmosphere problem for Lab 01 and all subparts.

TO REPRODUCE THE VALUES AND PLOPTS IN MY REPORT, DO THIS:
<tbd>
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

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


def question_3():
    '''
    This code answers Quetsion 3 from Lab. It contains 2 separate experiments and generate three figures.

    Returns:
        figs: Plot
            Show the plots for each of the experiment and question.
    
    To use this code, run figs = question_3()
    For displaying plot for experiment 1, run figs["exp1"].show()
    For displaying plot for experiment 2, run figs["exp2"].show()
    For displaying plot for Altitude vs Temprature, run figs["profile"].show()
    '''

    figs = {}
    # Experiment 1
    emissivities = np.linspace(0.05, 1.0, 40)
    Ts_vals = [n_layer_atmos(1, epsilon=eps)[0] for eps in emissivities]

    # find epsilon giving ~288 K
    idx = np.argmin(np.abs(np.array(Ts_vals) - 288))
    eps_best = emissivities[idx]
    Ts_best = Ts_vals[idx]

    fig1, ax1 = plt.subplots(1, 1, figsize=(8,8))

    ax1.plot(emissivities, Ts_vals, 'o-')
    ax1.axhline(288, color='k', linestyle='--', label="Target Ts=288K")
    ax1.axvline(eps_best, color='r', linestyle='--', label="Emissivity that yileds 288K")
    ax1.set_xlabel("Emissivity ε")
    ax1.set_ylabel("Surface Temperature Ts (K)")
    ax1.set_title("Experiment 1: 1 layer, Ts vs Emissivity")
    ax1.legend(loc='best')
    figs["exp1"] = fig1

    print(f"[Exp1] To reach ~288K with 1 layer, model requires ε ≈ {eps_best:.3f} (Ts={Ts_best:.2f} K).")

    # Experiment 2
    fixed_eps = 0.255
    layers = range(0, 15)
    Ts_layers = [n_layer_atmos(N, epsilon=fixed_eps)[0] for N in layers]

    idx2 = np.argmin(np.abs(np.array(Ts_layers) - 288))
    N_best = layers[idx2]
    Ts_best2 = Ts_layers[idx2]

    fig2, ax2 = plt.subplots(1, 1, figsize=(8,8))

    ax2.plot(layers, Ts_layers, 's-')
    ax2.axhline(288, color='k', linestyle='--', label="Target Ts=288K")
    ax2.axvline(N_best, color='r', linestyle='--', label="Number of layers that yield 288K")
    ax2.set_xlabel("Number of layers N")
    ax2.set_ylabel("Surface Temperature Ts (K)")
    ax2.set_title(f"Experiment 2: ε={fixed_eps}, Ts vs N")
    ax2.legend(loc='best')

    print(f"[Exp2] With ε={fixed_eps}, need N ≈ {N_best} layers to get Ts ≈ {Ts_best2:.2f} K.")
    figs["exp2"] = fig2

    temps_profile = n_layer_atmos(N_best, epsilon=fixed_eps)
    altitude = range(0, N_best+1)

    fig3, ax3 = plt.subplots(1, 1, figsize=(8,8))

    ax3.plot(temps_profile, altitude, 'o-')
    ax3.set_xlabel("Temperature (K)")
    ax3.set_ylabel("Altitude (layer)")
    ax3.set_title(f"Experiment 2: Altitude Profile (ε={fixed_eps}, N={N_best})")
    figs["profile"] = fig3

    return figs

def n_layer_atmos_venus(target_ts=700, epsilon=1, albedo=0.75, s0=2600):
    '''
    Solve the N-layer atmosphere problem.

    Parameters:
        target_ts : int
            The target surfave temperature of Venus, defaulted to 700.
        epsilon : float
            Emissivity of each layer, defaulted to 1.
        albedo : float
            Planetary albedo, defaulted to 1.
        s0 : float
            Solar irradiance [W/m^2], defaulted to 1350.
        debug : bool
            If True, print matrices.
    
    Returns:
        N_best : int
            Smallest number of layers giving Ts >= target_ts.
    '''

    Ts_list = []
    N_best = None

    for N in range(1, 100):
        temps = n_layer_atmos(N, epsilon=epsilon, albedo=albedo, s0=s0)
        Ts = temps[0]
        Ts_list.append(Ts)
        if N_best is None and Ts >= target_ts:
            N_best = N

    # Plot results
    plt.figure()
    plt.plot(range(1, len(Ts_list)+1), Ts_list, 'o-', label="Model Ts")
    plt.axhline(target_ts, color='k', linestyle='--', label=f"Target Ts={target_ts} K")
    if N_best is not None:
        plt.axvline(N_best, color='r', linestyle='--', label=f"N ≈ {N_best}")
    plt.xlabel("Number of layers N")
    plt.ylabel("Surface temperature Ts (K)")
    plt.title(f"Venus: Surface temperature vs Number of layers)")
    plt.legend()
    plt.show()

    print(f"To reach ~{target_ts} K, Venus requires about N = {N_best} layers.")

    return N_best

def nuclear_winter(nlayers=5, epsilon=0.5, albedo=0.33, s0=1350):
    '''
    Solve the N-layer atmosphere problem.
    To use this, just run nuclear_winter(), and the plot will pop up.

    Parameters:
        nlayers : int
            Number of atmospheric layers, defaulted to 5 under this scenario.
        epsilon : float
            Emissivity of each layer, defaulted to 0.5.
        albedo : float
            Planetary albedo, defaulted to 0.33.
        s0 : float
            Solar irradiance, defaulted to 1350.
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
    b[0] = 0.0
    b[nlayers] = -0.25 * s0 * (1-albedo)

    # Invert matrix:
    Ainv = np.linalg.inv(A) 
    # Get solution:
    fluxes = np.matmul(Ainv, b)

    temps = np.zeros_like(fluxes)
    temps[0] = np.power((fluxes[0] / sigma), 1/4)
    temps[1:] = np.power(fluxes[1:] / (sigma * epsilon), 1/4)

    altitudes = np.arange(0, len(temps))
    plt.plot(temps, altitudes, 'o-')
    plt.xlabel("Temperature (K)")
    plt.ylabel("Altitude (layer)")
    plt.title("Nuclear Winter: 5 layers, ε=0.5, S0=1350")
    plt.grid(True)
    plt.show()

