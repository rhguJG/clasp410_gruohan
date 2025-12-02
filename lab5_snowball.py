#!/usr/bin/env python3

'''
Lab 5: Snowball Earth.
For question 1, run problem1(); plt.show()
For question 2, run problem2(), then run problem2_sensitivity(best_lam=31.58, best_emiss=0.71)
For question 3, run problem3(31.58,0.71)
For question 4, run problem4(lam=31.58, emiss=0.71)
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Some constants:
radearth = 6357000.  # Earth radius in meters.
mxdlyr = 50.         # depth of mixed layer (m)
sigma = 5.67e-8      # Steffan-Boltzman constant
C = 4.2e6            # Heat capacity of water
rho = 1020           # Density of sea-water (kg/m^3)


def gen_grid(npoints=18):
    '''
    Create a evenly spaced latitudinal grid with `npoints` cell centers.
    Grid will always run from zero to 180 as the edges of the grid. This
    means that the first grid point will be `dLat/2` from 0 degrees and the
    last point will be `180 - dLat/2`.

    Parameters
    ----------
    npoints : int, defaults to 18
        Number of grid points to create.

    Returns
    -------
    dLat : float
        Grid spacing in latitude (degrees)
    lats : numpy array
        Locations of all grid cell centers.
    '''

    dlat = 180 / npoints  # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints)  # Lat cell centers.

    return dlat, lats


def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''
    # Set initial temperature curve:
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    # Get base grid:
    npoints = T_warm.size
    dlat, lats = gen_grid(npoints)

    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp


def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation


def snowball_earth(nlat=18, tfinal=10000, dt=1.0, lam=100., emiss=1.0,
                   init_cond=temp_warm, apply_spherecorr=False, albice=.6,
                   albgnd=.3, apply_insol=False, solar=1370):
    '''
    Solve the snowball Earth problem.

    Parameters
    ----------
    nlat : int, defaults to 18
        Number of latitude cells.
    tfinal : int or float, defaults to 10,000
        Time length of simulation in years.
    dt : int or float, defaults to 1.0
        Size of timestep in years.
    lam : float, defaults to 100
        Set ocean diffusivity
    emiss : float, defaults to 1.0
        Set emissivity of Earth/ground.
    init_cond : function, float, or array
        Set the initial condition of the simulation. If a function is given,
        it must take latitudes as input and return temperature as a function
        of lat. Otherwise, the given values are used as-is.
    apply_spherecorr : bool, defaults to False
        Apply spherical correction term
    apply_insol : bool, defaults to False
        Apply insolation term.
    solar : float, defaults to 1370
        Set level of solar forcing in W/m2
    albice, albgnd : float, defaults to .6 and .3
        Set albedo values for ice and ground.

    Returns
    --------
    lats : Numpy array
        Latitudes representing cell centers in degrees; 0 is south pole
        180 is north.
    Temp : Numpy array
        Temperature as a function of latitude.
    '''

    # Set up grid:
    dlat, lats = gen_grid(nlat)
    # Y-spacing for cells in physical units:
    dy = np.pi * radearth / nlat

    # Create our first derivative operator.
    B = np.zeros((nlat, nlat))
    B[np.arange(nlat-1)+1, np.arange(nlat-1)] = -1
    B[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    B[0, :] = B[-1, :] = 0

    # Create area array:
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    # Get derivative of Area:
    dAxz = np.matmul(B, Axz)

    # Set number of time steps:
    nsteps = int(tfinal / dt)

    # Set timestep to seconds:
    dt = dt * 365 * 24 * 3600

    # Create insolation:
    insol = insolation(solar, lats)

    # Create temp array; set our initial condition
    Temp = np.zeros(nlat)
    if callable(init_cond):
        Temp = init_cond(lats)
    else:
        Temp += init_cond

    # Create our K matrix:
    K = np.zeros((nlat, nlat))
    K[np.arange(nlat), np.arange(nlat)] = -2
    K[np.arange(nlat-1)+1, np.arange(nlat-1)] = 1
    K[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    # Boundary conditions:
    K[0, 1], K[-1, -2] = 2, 2
    # Units!
    K *= 1/dy**2

    # Create L matrix.
    Linv = np.linalg.inv(np.eye(nlat) - dt * lam * K)

    # Set initial albedo.
    albedo = np.zeros(nlat)
    loc_ice = Temp <= -10  # Sea water freezes at ten below.
    albedo[loc_ice] = albice
    albedo[~loc_ice] = albgnd

    # SOLVE!
    for istep in range(nsteps):
        # Update Albedo:
        loc_ice = Temp <= -10  # Sea water freezes at ten below.
        albedo[loc_ice] = albice
        albedo[~loc_ice] = albgnd

        # Create spherical coordinates correction term
        if apply_spherecorr:
            sphercorr = (lam*dt) / (4*Axz*dy**2) * np.matmul(B, Temp) * dAxz
        else:
            sphercorr = 0

        # Apply radiative/insolation term:
        if apply_insol:
            radiative = (1-albedo)*insol - emiss*sigma*(Temp+273)**4
            Temp += dt * radiative / (rho*C*mxdlyr)

        # Advance solution.
        Temp = np.matmul(Linv, Temp + sphercorr)

    return lats, Temp


def problem1():
    '''
    Create solution figure for Problem 1 (also validate our code qualitatively)
    '''

    # Get warm Earth initial condition.
    dlat, lats = gen_grid()
    temp_init = temp_warm(lats)

    # Get solution after 10K years for each combination of terms:
    lats, temp_diff = snowball_earth()
    lats, temp_sphe = snowball_earth(apply_spherecorr=True)
    lats, temp_alls = snowball_earth(apply_spherecorr=True, apply_insol=True,
                                     albice=.3)

    # Create a fancy plot!
    fig, ax = plt.subplots(1, 1)
    ax.plot(lats-90, temp_init, label='Initial Condition')
    ax.plot(lats-90, temp_diff, label='Diffusion Only')
    ax.plot(lats-90, temp_sphe, label='Diffusion + Spherical Corr.')
    ax.plot(lats-90, temp_alls, label='Diffusion + Spherical Corr. + Radiative')

    # Customize like those annoying insurance commercials
    ax.set_title('Solution after 10,000 Years')
    ax.set_ylabel(r'Temp ($^{\circ}C$)')
    ax.set_xlabel('Latitude')
    ax.legend(loc='best')


def test_functions():
    '''Test our functions'''

    print('Test gen_grid')
    print('For npoints=5:')
    dlat_correct, lats_correct = 36.0, np.array([18., 54., 90., 126., 162.])
    result = gen_grid(5)
    if (result[0] == dlat_correct) and np.all(result[1] == lats_correct):
        print('\tPassed!')
    else:
        print('\tFAILED!')
        print(f"Expected: {dlat_correct}, {lats_correct}")
        print(f"Got: {gen_grid(5)}")

def problem2():
    '''
    Tune the model parameters (diffusivity and emissivity) to match
    the modern warm Earth temperature profile.
    '''

    # Get grid and the target data.
    dlat, lats = gen_grid()
    temp_target = temp_warm(lats)

    # Define the parameter space to search.
    lam_values = np.linspace(0, 150, 20)      
    emiss_values = np.linspace(0.0, 1.0, 8)  

    # List to store our error metrics.
    results = []

    for lam in lam_values:
        for emiss in emiss_values:

            # Run the model with current parameter guess.
            _, temp_model = snowball_earth(
                lam=lam,
                emiss=emiss,
                apply_spherecorr=True,
                apply_insol=True,
                albice=.3, albgnd=.3  # Constant albedo for tuning
            )

            # Calculate the Root Mean Square Error (RMSE).
            rmse = np.sqrt(np.mean((temp_model - temp_target)**2))

            results.append((rmse, lam, emiss))

    # Sort results by error and pick the winner.
    results.sort(key=lambda x: x[0])
    best_rmse, best_lam, best_emiss = results[0]

    print("Problem 2 Results")
    print(f"Best diffusivity λ: {best_lam:.2f}")
    print(f"Best emissivity ε: {best_emiss:.2f}")
    print(f"RMSE: {best_rmse:.4f}")

    # Run one last time with best params to make a plot.
    _, best_temp = snowball_earth(
        lam=best_lam,
        emiss=best_emiss,
        apply_spherecorr=True,
        apply_insol=True,
        albice=.3, albgnd=.3
    )

    plt.figure()
    plt.plot(lats-90, temp_target, label='Target Warm Earth')
    plt.plot(lats-90, best_temp, '--', label='Best-fit Model')
    plt.xlabel('Latitude')
    plt.ylabel('Temperature (°C)')
    plt.title('Best Parameter Fit')
    plt.legend()
    plt.show()

def problem2_sensitivity(best_lam=100.0, best_emiss=1.0):
    '''
    Investigate how individual parameters (diffusivity and emissivity)
    alter the temperature profile shape and magnitude.
    
    Parameters
    ----------
    best_lam : float, optional 
        The tuned diffusivity value (default=100.0).
    best_emiss : float, optional 
        The tuned emissivity value (default=1.0).
    '''
    dlat, lats = gen_grid()
    temp_target = temp_warm(lats)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Test sensitivity to Diffusivity (Lambda).
    test_lams = [0, 50, 150] 
    ax1.plot(lats-90, temp_target, 'k--', label='Target (Warm Earth)')
    
    for lam in test_lams:
        _, temp = snowball_earth(lam=lam, emiss=best_emiss, 
                                 apply_spherecorr=True, apply_insol=True, 
                                 albice=.3, albgnd=.3)
        ax1.plot(lats-90, temp, label=rf'$\lambda={lam}$')
    
    # Label the first panel.
    ax1.set_title(rf'Sensitivity to Diffusivity ($\epsilon={best_emiss}$)')
    ax1.set_xlabel('Latitude')
    ax1.set_ylabel('Temp (C)')
    ax1.legend()
    
    # Test sensitivity to Emissivity.
    test_emiss = [0.4, 0.7, 1.0] 
    ax2.plot(lats-90, temp_target, 'k--', label='Target')
    
    for em in test_emiss:
        _, temp = snowball_earth(lam=best_lam, emiss=em, 
                                 apply_spherecorr=True, apply_insol=True, 
                                 albice=.3, albgnd=.3)
        ax2.plot(lats-90, temp, label=rf'$\epsilon={em}$')
        
    # Label the second panel.
    ax2.set_title(rf'Sensitivity to Emissivity ($\lambda={best_lam}$)')
    ax2.set_xlabel('Latitude')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def problem3(lam=50., emiss=0.80):
    '''
    Test the stability of climate states under extreme initial conditions
    (Hot Start vs Cold Start) and a Flash Freeze scenario.
    '''
    # Get grid.
    dlat, lats = gen_grid()

    # Define extreme initial states.
    hot_init  = np.full_like(lats,  60.0, dtype=float)
    cold_init = np.full_like(lats, -60.0, dtype=float)

    # Run model starting from Hot Earth.
    _, T_hot = snowball_earth(
        lam=lam, emiss=emiss,
        init_cond=hot_init,
        apply_spherecorr=True, apply_insol=True,
        albice=0.6, albgnd=0.3
    )

    # Run model starting from Cold Earth.
    _, T_cold = snowball_earth(
        lam=lam, emiss=emiss,
        init_cond=cold_init,
        apply_spherecorr=True, apply_insol=True,
        albice=0.6, albgnd=0.3
    )

    # Create a Flash Freeze scenario.
    # First, spin up a warm steady state.
    _, T_warm_steadystate = snowball_earth(
        lam=lam, emiss=emiss,
        init_cond=temp_warm,                 
        apply_spherecorr=True, apply_insol=True,
        albice=0.3, albgnd=0.3               
    )
    # Then force albedo to ice everywhere instantly.
    _, T_flash = snowball_earth(
        lam=lam, emiss=emiss,
        init_cond=T_warm_steadystate,        
        apply_spherecorr=True, apply_insol=True,
        albice=0.6, albgnd=0.6               
    )

    # Calculate area-weighted global means.
    weights = np.sin(np.radians(lats))       
    def global_mean(T): 
        return np.average(T, weights=weights)

    print("Problem 3 (Equilibrium Global-Mean Temp, °C)")
    print(f"Hot Earth init  : {global_mean(T_hot):6.2f}")
    print(f"Cold Earth init : {global_mean(T_cold):6.2f}")
    print(f"Flash freeze    : {global_mean(T_flash):6.2f}")

    # Plot the resulting equilibrium profiles.
    plt.figure()
    plt.plot(lats-90, T_hot,  label='Hot init (+60°C), dynamic albedo')
    plt.plot(lats-90, T_cold, label='Cold init (-60°C), dynamic albedo')
    plt.plot(lats-90, T_flash,label='Flash-freeze (albedo set to 0.6)')
    plt.xlabel('Latitude')
    plt.ylabel('Temperature (°C)')
    plt.title('Equilibrium Profiles')
    plt.legend()
    plt.show()

def problem4(lam=50., emiss=0.80, S0=1370., g0=0.4, g1=1.4, dg=0.05):
    '''
    Explore the hysteresis loop by slowly varying the solar luminosity
    multiplier (gamma) forward and backward.
    '''
    # Set up grid and weighting factors.
    dlat, lats = gen_grid()
    weights = np.sin(np.radians(lats))
    def global_mean(T): return np.average(T, weights=weights)

    # Helper function to run a specific solar forcing state.
    def run_gamma(gamma, init_T):
        _, T = snowball_earth(
            lam=lam, emiss=emiss,
            init_cond=init_T,
            apply_spherecorr=True, apply_insol=True,
            albice=0.6, albgnd=0.3,        
            solar=S0 * gamma               
        )
        return T

    # Perform the Forward sweep (increasing solar flux).
    gam_fwd = np.round(np.arange(g0, g1 + 1e-9, dg), 3)
    means_fwd = []
    
    # Start cold.
    T_prev = np.full_like(lats, -60.0, dtype=float)
    for g in gam_fwd:
        T_prev = run_gamma(g, T_prev)
        means_fwd.append(global_mean(T_prev))

    # Perform the Backward sweep (decreasing solar flux).
    gam_bwd = np.round(np.arange(g1, g0 - 1e-9, -dg), 3)
    means_bwd = []
    
    # Continue from the last hot state.
    for g in gam_bwd:
        T_prev = run_gamma(g, T_prev)
        means_bwd.append(global_mean(T_prev))

    # Visualize the hysteresis loop.
    plt.figure()
    plt.plot(gam_fwd, means_fwd, label='Forward (g ↑)', linewidth=3)
    plt.plot(gam_bwd, means_bwd, '--', label='Backward (g ↓)', linewidth=3)
    plt.xlabel(r'Solar multiplier $\gamma$')
    plt.ylabel('Global-mean Temperature (°C)')
    plt.title('Hysteresis of Global Temperature vs Solar Forcing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Print summary stats.
    print("Problem 4 summary")
    print(f"Min/Max forward mean T: {min(means_fwd):.2f} / {max(means_fwd):.2f} °C")
    print(f"Min/Max backward mean T: {min(means_bwd):.2f} / {max(means_bwd):.2f} °C")
    
    return (gam_fwd, np.array(means_fwd)), (gam_bwd, np.array(means_bwd))