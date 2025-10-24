#!/usr/bin/env python3

'''
Tools and methods for completing Lab 3 which is the best lab.

Usage
-----
For Q1, please run:
    t,x,U=solve_heat(xstop=1, tstop=0.2, dx=0.2, dt=0.02, c2=1, lowerbound=0, upperbound=0, question=1)
    np.allclose(U, sol10p3, atol=1e-12)
    The result should be true.
For Q2:
    To generate Fig 1 & 2, please run:
    t, x, U, active_layer_depth, permafrost_depth, temp_change, years = Q2(show_plot=True)
    To generate Fig 3, please run:
    t, x, U, active_layer_depth, permafrost_depth, temp_change, years = Q2(years = 65, show_plot=True)
    To generate Fig 4, please run:
    t, x, U, active_layer_depth, permafrost_depth, temp_change, years = Q2(years = 100, show_plot=True)
For Q3, please run:
    shifts, actives, pfs, years_used=Q3(start_years=65, tol=0.01, max_years=150, show_any_plots=False)
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Solution to problem 10.3
sol10p3 = [[0.000000, 0.640000, 0.960000, 0.960000, 0.640000, 0.000000],
           [0.000000, 0.480000, 0.800000, 0.800000, 0.480000, 0.000000],
           [0.000000, 0.400000, 0.640000, 0.640000, 0.400000, 0.000000],
           [0.000000, 0.320000, 0.520000, 0.520000, 0.320000, 0.000000],
           [0.000000, 0.260000, 0.420000, 0.420000, 0.260000, 0.000000],
           [0.000000, 0.210000, 0.340000, 0.340000, 0.210000, 0.000000],
           [0.000000, 0.170000, 0.275000, 0.275000, 0.170000, 0.000000],
           [0.000000, 0.137500, 0.222500, 0.222500, 0.137500, 0.000000],
           [0.000000, 0.111250, 0.180000, 0.180000, 0.111250, 0.000000],
           [0.000000, 0.090000, 0.145625, 0.145625, 0.090000, 0.000000],
           [0.000000, 0.072812, 0.117813, 0.117813, 0.072812, 0.000000]]
sol10p3 = np.array(sol10p3).transpose()


def solve_heat(xstop=1, tstop=0.2, dx=0.2, dt=0.02, c2=1, lowerbound=0,
               upperbound=0, question=1):
    '''
    A function for solving the heat equation.
    Apply Neumann boundary conditions such that dU/dx = 0.

    Parameters
    ----------
    xstop : float
        Domain depth in meters. Space is [0, xstop].
    tstop : float
        Total run time in seconds. Time is [0, tstop].
    dx : float
        Space step in meters.
    dt : float
        Time step in seconds.
    c2 : float
        c^2, the square of the diffusion coefficient.
    upperbound, lowerbound : None, scalar, or func
        Set the lower and upper boundary conditions. If either is set to
        None, then Neumann boundary condtions are used and the boundary value
        is set to be equal to its neighbor, producing zero gradient.
        Otherwise, Dirichlet conditions are used and either a scalar constant
        is provided or a function should be provided that accepts time and
        returns a value.
    question : int
        Chooses the initial condition. 1 uses 4x - 4x^2. 2 uses zeros.

    Returns
    -------
    x, t : 1D Numpy arrays
        Space and time values, respectively.
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime
    '''


    # Check our stability criterion:
    dt_max = dx**2 / (2*c2)
    if dt > dt_max:
        raise ValueError(f'DANGER: dt={dt} > dt_max={dt_max}.')

    # Get grid sizes (plus one to include "0" as well.)
    N = int(tstop / dt) + 1
    M = int(xstop / dx) + 1

    # Set up space and time grid:
    t = np.linspace(0, tstop, N)
    x = np.linspace(0, xstop, M)

    # Create solution matrix; set initial conditions
    U = np.zeros([M, N])
    if (question == 1):
         U[:, 0] = 4*x - 4*x**2
    elif (question == 2):
        U[:, 0 ] = 0

    # Get our "r" coeff:
    r = c2 * (dt/dx**2)

    # Solve our equation!
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])

        # Apply boundary conditions:
        # Lower boundary
        if lowerbound is None:  # Neumann
            U[0, j+1] = U[1, j+1]
        elif callable(lowerbound):  # Dirichlet/constant
            U[0, j+1] = lowerbound(t[j+1])
        else:
            U[0, j+1] = lowerbound

        # Upper boundary
        if upperbound is None:  # Neumann
            U[-1, j+1] = U[-2, j+1]
        elif callable(upperbound):  # Dirichlet/constant
            U[-1, j+1] = upperbound(t[j+1])
        else:
            U[-1, j+1] = upperbound


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
    ax.invert_yaxis()

    fig.tight_layout()

    return fig, ax, cbar

# Kangerlussuaq average temperature:
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4, 10.7, 8.5, 3.1, -6.0, -12.0, -16.9])

def temp_kanger(t): 
    '''
    For an array of times in days, return timeseries of temperature for Kangerlussuaq, Greenland.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()


def Q2(years=5, temp_shift=0, show_plot=False):
    '''
    Question 2: Investigate permafrost in Kangerlussuaq, Greenland.
    Solves U_t = c^2 U_xx from the surface (x = 0 m) to 100 m depth for a
    given number of years. The surface boundary (x = 0) follows the
    Kangerlussuaq annual temperature cycle plus an optional constant shift
    for warming scenarios. The deep boundary (x = 100 m) is fixed at 5 °C.
    The initial condition for this lab question is U(x, 0) = 0 °C.
    
    Parameters
    ----------
    years : int, default 5
        Number of years to simulate
    temp_shift : float, default 0
        Temperature shift in °C for global warming scenarios (Q3)
        Default 0 for baseline (Q2)
    show_plot : bool, default False
        If True make a heatmap of U(x, t) and a final-year winter and summer
        profile with guides at 0 °C the active layer depth and the bottom of
        permafrost. If False, show nothing.
    
    Returns
    -------
    t, x, U : arrays
        Time, space, and temperature solution
    active_layer_depth : float
        Depth of active layer in meters
    permafrost_depth : float
        Thickness of permafrost layer in meters
    temp_change : float
        Absolute change in mean deep-zone temperature between the last two years 
        in °C used as a simple steady-state test
    years : int
        The number of years actually simulated which equals the input years
    '''
    
    # Parameters
    xstop = 100 
    dx = 0.5
    c2 = 0.25e-6
    
    # Calculate stable time step
    dt_max = dx**2 / (2 * c2)
    dt = 0.5 * dt_max
    
    # Run for multiple years to reach steady state
    tstop = years * 365 * 24 * 3600  # convert years to seconds
    
    # Upper boundary: surface temperature varies with Kangerlussuaq climate
    # temp_kanger expects time in days
    def lower_bc(t_seconds):
        t_days = t_seconds / (24 * 3600)
        return temp_kanger(t_days) + temp_shift
      
    # Solve the heat equation
    t, x, U = solve_heat(
        xstop=100,
        tstop=tstop,
        dx=0.5,
        dt=dt,
        c2=c2,
        lowerbound=lower_bc,
        upperbound=5.0,
        question=2
    )
    
    # Convert time to years for plotting
    t_years = t / (365 * 24 * 3600)
    
    # Extract seasonal profiles from final year
    loc = int(-365 * 24 * 3600 / dt)
    winter = U[:, loc:].min(axis=1)
    summer = U[:, loc:].max(axis=1)

    # Check for steady state in isothermal zone (deep region, e.g., 60-100m)
    isothermal_idx_start = int(60 / dx)  # Start at 60m depth
    loc_last_year = int(-365 * 24 * 3600 / dt)
    loc_prev_year = int(-2 * 365 * 24 * 3600 / dt)

    temp_last_year = U[isothermal_idx_start:, loc_last_year:].mean()
    temp_prev_year = U[isothermal_idx_start:, loc_prev_year:loc_last_year].mean()
    temp_change = abs(temp_last_year - temp_prev_year)
    is_steady = temp_change < 0.01
    
    print(f"Isothermal zone temp change (last 2 years): {temp_change:.4f}°C")
    if is_steady:
        print("System has reached steady state")
    else:
        print("It's not steady yet. Run longer.")
    
    # Find active layer depth (where summer temp > 0°C)
    idx = np.where((summer[:-1] > 0.0) & (summer[1:] <= 0.0))[0]
    if idx.size:
        i = idx[0]
        x0, x1 = x[i], x[i+1]
        y0, y1 = summer[i], summer[i+1]
        active_layer_depth = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
    else:
        active_layer_depth = xstop
    
    # Find permafrost layer depth (where winter temp goes above 0°C permanently)
    idxb = np.where((winter[:-1] <= 0.0) & (winter[1:] > 0.0))[0]
    if idxb.size:
        j = idxb[0]
        xb0, xb1 = x[j], x[j+1]
        yw0, yw1 = winter[j], winter[j+1]
        permafrost_bottom = xb0 + (0.0 - yw0) * (xb1 - xb0) / (yw1 - yw0)
    else:
        permafrost_bottom = xstop
    
    if (show_plot==True):
        # Create space-time heat map
        fig1, ax1, cbar1 = plot_heatsolve(
            t_years, x, U,
            title='Kangerlussuaq Ground Temperature',
            cmap='seismic',
            vmin=-25, vmax=25
        )
        ax1.set_xlabel('Time (years)')
        ax1.set_ylabel('Depth (m)')
        
        # Temperature profile
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        ax2.plot(winter, x, 'b-', linewidth=2, label='Winter')
        ax2.plot(summer, x, 'r--', linewidth=2, label='Summer')

        # Vertical line at 0 °C
        ax2.axvline(0.0, color='k', linewidth=1, alpha=0.7)

        # Horizontal lines: active-layer depth and permafrost bottom
        ax2.axhline(active_layer_depth, color='g', linestyle=':', linewidth=1.8,
                    label=f'Active layer depth ≈ {active_layer_depth:.1f} m')
        ax2.axhline(permafrost_bottom, color='purple', linestyle='--', linewidth=1.8,
                    label=f'Permafrost bottom ≈ {permafrost_bottom:.1f} m')
        
        ax2.set_xlabel('Temperature (°C)', fontsize=12)
        ax2.set_ylabel('Depth (m)', fontsize=12)
        ax2.set_title('Ground Temperature Profile')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-8,6)
        ax2.set_ylim(100,0)
        fig2.tight_layout()
        
        plt.show()

    permafrost_depth = permafrost_bottom - active_layer_depth
    print(f"Active layer depth {active_layer_depth:.1f}m, permafrost thickness {permafrost_depth:.1f}m")
    
    return t, x, U, active_layer_depth, permafrost_depth, temp_change, years

def Q3(start_years=63, tol=0.01, max_years=150, show_any_plots=False):
    """
    Run Q3 for +0.5, +1.0, +3.0 °C.
    For each case, extend the run until the deep-zone change is below tol.
    Print sentences and make one summary plot.

    Parameters
    ----------
    start_years : int, default 63
        Initial run length in years for each scenario. The function extends
        this year by year until the steady test passes or max_years is hit.
    tol : float, default 0.01
        Steady test threshold in degrees C for the mean deep zone change
        between the last two simulated years.
    max_years : int, default 150
        Maximum years allowed for any scenario.
    show_any_plots : bool, default False
        Reserved flag. Currently plots are always shown for the summary.

    Returns
    -------
    shifts : list of float
        The temperature shifts in degrees C used for Q3.
    actives : list of float
        Active layer depths in meters at steady for each shift.
    pfs : list of float
        Permafrost thicknesses in meters at steady for each shift.
    years_used : list
        Placeholder list for symmetry with other helpers. Not populated here.
    """
    shifts = [0.5, 1.0, 3.0]
    actives, pfs, years_used = [], [], []

    for s in shifts:
        yrs = start_years
        while True:
            _, _, _, a, pf, change, used = Q2(years=yrs, temp_shift=s)
            if change < tol or yrs >= max_years:
                print(f"For a temperature shift of {s:.1f} degrees C, the active layer depth is {a:.1f} m and the permafrost thickness is {pf:.1f} m. Final year used {used}.")
                actives.append(a)
                pfs.append(pf)
                break
            yrs += 1

    # one summary plot 
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(shifts, actives, "bo-", label="Active layer depth")
    ax.plot(shifts, pfs, "rs--", label="Permafrost thickness")

    # Add lines for active layer depth
    for x, y in zip(shifts, actives):
        ax.annotate(f"({x:.1f}, {y:.1f})", xy=(x, y), xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=11, color="blue")

    # Add lines for permafrost layer bottom
    for x, y in zip(shifts, pfs):
        ax.annotate(f"({x:.1f}, {y:.1f})", xy=(x, y), xytext=(0, -14), textcoords="offset points",
            ha="center", fontsize=11, color="red")

    ax.set_xlabel("Temperature shift (°C)", fontsize=12)
    ax.set_ylabel("Depth or thickness (m)", fontsize=12)
    ax.set_title("Impact of global warming on permafrost")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    return shifts, actives, pfs, years_used
