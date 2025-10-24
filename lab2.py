#!/usr/bin/env python3
'''
Lab 2: Population Control

This script models Lab 2, solves Lotka-Volterra equations
for competition and predator-prey systems. Both Euler and
RK8 solvers are used.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.style.use("seaborn-v0_8")


## Derivative functions


def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    """
    Competition model ODEs. Two species competing.

    Inputs:
        t : time (float)
        N : [N1, N2] populations
        a, b, c, d : model parameters
    Outputs:
        [dN1dt, dN2dt]
    """
    # Unpack species
    N1, N2 = N
    # Logistic growth - competition terms
    dN1dt = a * N1 * (1 - N1) - b * N1 * N2
    dN2dt = c * N2 * (1 - N2) - d * N1 * N2
    return [dN1dt, dN2dt]


def dNdt_predprey(t, N, a=1, b=2, c=1, d=3):
    """
    Predator-prey model ODEs. Prey growth and predator hunting them.

    Inputs:
        t : time (float)
        N : [N1, N2] populations
        a, b, c, d : model parameters
    Outputs:
        [dN1dt, dN2dt]
    """
    # Unpack species
    N1, N2 = N
    # Prey grows, eaten by predator
    dN1dt = a * N1 - b * N1 * N2
    # Predator dies, grows when eating prey
    dN2dt = -c * N2 + d * N1 * N2
    return [dN1dt, dN2dt]


## Euler solver


def euler_solve(func, N1_init=0.3, N2_init=0.6, dT=0.1, t_final=100.0, a=1, b=2, c=1, d=3):
    """
    Euler solver (fixed step). Models the populations step by step.

    Inputs:
        func : derivative function
        N1_init, N2_init : initial populations
        dT : step size (years)
        t_final : total time (years)
        a, b, c, d : model parameters
    Outputs:
        time : array of times
        N1 : array of N1 values
        N2 : array of N2 values
    """
    # Build time array
    time = np.arange(0, t_final + dT, dT)
    # Storage arrays
    N1 = np.zeros_like(time)
    N2 = np.zeros_like(time)
    # Set initial values
    N1[0], N2[0] = N1_init, N2_init

    # March forward in time
    for i in range(1, len(time)):
        dN1dt, dN2dt = func(time[i-1], [N1[i-1], N2[i-1]], a, b, c, d)
        # Euler update: new = old + step * slope
        N1[i] = N1[i-1] + dT * dN1dt
        N2[i] = N2[i-1] + dT * dN2dt
    return time, N1, N2


## RK8 solver


def solve_rk8(func, N1_init=0.3, N2_init=0.6, dT=10, t_final=100.0, a=1, b=2, c=1, d=3):
    """
    RK8 solver (DOP853, adaptive step). Models smoother populations and takes smaller steps when necessary.

    Inputs:
        func : derivative function
        N1_init, N2_init : initial populations
        dT : max step size (years)
        t_final : total time (years)
        a, b, c, d : model parameters
    Outputs:
        time : array of times
        N1 : array of N1 values
        N2 : array of N2 values
    """
    # Call SciPy ODE solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                       args=(a, b, c, d), method="DOP853", max_step=dT)
    return result.t, result.y[0], result.y[1]


## LAB QUESTIONS
def question_1(dt_comp, dt_pred, max_step=10):
    '''
    Reproduce the lab's **Question 1** comparison figure.

    Runs both solvers on the competition model (Euler step `dt_comp`)
    and on the predator-prey model (Euler step `dt_pred`) for 100 years,
    and plots side-by-side time series with RK8 overlays.

    Parameters
    ----------
    dt_comp : float
        Euler time step for the competition model.
    dt_pred : float
        Euler time step for the predator-prey model.
    max_step : float, optional
        RK8 for both panels.

    Returns
    -------
    None
        Displays the two-panel figure.

    Examples Usage
    --------
    # Lab default
    question_1(dt_comp=1.0, dt_pred=0.05, max_step=10)
    # Coarser Euler to demonstrate instability
    question_1(dt_comp=2.0, dt_pred=0.20, max_step=10)
    # Finer Euler to reduce error
    question_1(dt_comp=0.2, dt_pred=0.01, max_step=10)
    For the graph only shows N_1, comment line 179, 180 and 182 and run this code again.
    question_1(dt_comp=0.2, dt_pred=0.01, max_step=10)
    '''
    N = [0.3, 0.6]
    Tfinal = 100

    # Competition
    tE_c, N1E_c, N2E_c = euler_solve(dNdt_comp, *N, dT=dt_comp, t_final=Tfinal)
    tR_c, N1R_c, N2R_c = solve_rk8(dNdt_comp, *N, dT=max_step, t_final=Tfinal)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.plot(tE_c, N1E_c, label=r'$N_1$ Euler', lw=2, color='blue')
    ax.plot(tE_c, N2E_c, label=r'$N_2$ Euler', lw=2, color='red')
    ax.plot(tR_c, N1R_c, linestyle=':', lw=3, color='blue', label=r'$N_1$ RK8')
    ax.plot(tR_c, N2R_c, linestyle=':', lw=3, color='red', label=r'$N_2$ RK8')
    ax.set_title("Lotka-Volterra Competition Model")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Population/Carrying Cap.")
    ax.legend(loc='upper left', frameon=True)
    ax.text(0.98, -0.12, "Coefficients: a=1, b=2, c=1, d=3",
            ha='right', va='top', transform=ax.transAxes)


    # Predator–Prey
    tE_p, N1E_p, N2E_p = euler_solve(dNdt_predprey, *N, dT=dt_pred, t_final=Tfinal)
    tR_p, N1R_p, N2R_p = solve_rk8(dNdt_predprey, *N, dT=max_step, t_final=Tfinal)

    # Plot
    ax = axes[1]
    ax.plot(tE_p, N1E_p, label=r'$N_1$ (Prey) Euler', lw=2, color='blue')
    ax.plot(tE_p, N2E_p, label=r'$N_2$ (Predator) Euler', lw=2, color='red')
    ax.plot(tR_p, N1R_p, linestyle=':', lw=3, color='blue', label=r"$N_1$ (Prey) RK8")
    ax.plot(tR_p, N2R_p, linestyle=':', lw=3, color='red', label=r"$N_2$ (Predator) RK8")  
    ax.set_title("Lotka-Volterra Predator-Prey Model")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Population/Carrying Cap.")
    ax.legend(loc='upper left', frameon=True)

    fig.tight_layout()

    plt.show()

def competition_plot(a, b, c, d, n=(0.3, 0.6), T=100, dt=0.02):
    """
    Plot competition time series for multiple parameter/initial condition sets on one axes.

    Parameters
    ----------
    a, b, c, d : sequence[float]
        Lists (or tuples) of parameters, one set per curve group.
    n : sequence[tuple(float, float)]
        List of initial conditions `[(N1_0, N2_0), ...]`, same length as `a`.
    T : float, optional
        Total integration time (years).
    dt : float, optional
        Euler step (years).

    Returns
    -------
    None
        Displays a single figure with paired N_1 and N_2 lines per case.
        Each pair shares a color; legend shows the initial values.
    
    Examples Usage
    --------
    To reproduce plot 7, run:
    a=[1.0,1.0,1.0]
    b=[0.5,0.5,0.5]
    c=[1.0,1.0,1.0]
    d=[0.6,0.6,0.6]
    n=[(0.2,0.8),(0.3,0.6),(0.8,0.2)]
    competition_plot(a, b, c, d, n, T=100, dt=0.02)

    To reproduce plot 8, run:
    a=[1.2,1.0,1.0]
    b=[0.6,0.3,0.95]
    c=[0.8,1.5,1.0]
    d=[0.4,0.9,0.9]
    n=[(0.3,0.6),(0.3,0.6),(0.3,0.6)]
    competition_plot(a, b, c, d, n, T=100, dt=0.02)
    """
    for (ai, bi, ci, di, ni) in zip(a, b, c, d, n):
        N1_init, N2_init = ni
        t_e, N1_e, N2_e = euler_solve(dNdt_comp, N1_init, N2_init, dT=dt, t_final=T,
                                    a=ai, b=bi, c=ci, d=di)
        t_r, N1_r, N2_r = solve_rk8(dNdt_comp, N1_init, N2_init, dT=dt, t_final=T,
                                a=ai, b=bi, c=ci, d=di)


        # Plot
        line1, = plt.plot(t_e, N1_e, label=f'$N_1$={N1_init}', lw=2)
        color = line1.get_color()
        plt.plot(t_e, N2_e, label=f'$N_2$={N2_init} ', color = color, lw=2)
        # plt.plot(t_r, N1_r, linestyle=':', lw=3, label=r'$N_1$ RK8')
        # plt.plot(t_r, N2_r, linestyle=':', lw=3, label=r'$N_2$ RK8')
    plt.title("Lotka-Volterra Competition Model")
    plt.xlabel("Time (years)")
    plt.ylabel("Population/Carrying Cap.")
    plt.legend(loc='upper right', frameon=True)
    fig = plt.gcf()
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.text(0.98, 0.02, f"Coefficients: a={a}, b={b}, c={c}, d={d}",
            ha='right', va='bottom')

    plt.show()

def prey_and_predator_plot(a, b, c, d, n=(0.3, 0.6), T=100, dt=0.1, ics=None):
    """
    Plot predator and prey time series (Euler & RK8) and the **phase diagram**.

    Parameters
    ----------
    a, b, c, d : float
        Model coefficients.
    n : tuple(float, float), optional
        Initial condition `(N1_0, N2_0)` = (prey, predator).
    T : float, optional
        Total integration time (years).
    dt : float, optional
        Euler step for the time-series panel; also used as RK8 `max_step`.

    Returns
    -------
    None
        Shows two figures:
        (1) time series with Euler and RK8,
        (2) phase diagram with trajectories, start/end markers, and nullclines.

    Example Usage
    -------------
    To reproduce plot 9 & 10, run
    prey_and_predator_plot(1, 2, 1, 3, n=(0.30, 0.60), T=100, dt=0.02)
    To reproduce plot 11 & 12, run
    prey_and_predator_plot(1, 2, 1, 3, n=(0.60, 0.30), T=100, dt=0.02)
    To reproduce plot 13 & 14, run
    prey_and_predator_plot(1.4, 2, 1, 3, n=(0.30, 0.60), T=100, dt=0.02)
    To reproduce plot 15 & 16, run
    prey_and_predator_plot(1, 2, 1.5, 3, n=(0.30, 0.60), T=100, dt=0.02)

    To reproduce plot 17, run
    prey_and_predator_plot(1, 2, 1, 3, n=(0.30, 0.60), T=100, dt=0.02,ics=[(0.3, 0.3), (0.6, 0.3), (0.9, 0.3)])
    To plot plot 18, run
    prey_and_predator_plot(1, 2, 1, 3, n=(0.30, 0.60), T=100, dt=0.02,ics=[(0.3, 0.3), (0.3, 0.6), (0.3, 0.9)])
    """
    N1_init, N2_init = n
    t_e, N1_e, N2_e = euler_solve(dNdt_predprey, N1_init, N2_init, dT=dt, t_final=T,
                                a=a, b=b, c=c, d=d)
    t_r, N1_r, N2_r = solve_rk8(dNdt_predprey, N1_init, N2_init, dT=dt, t_final=T,
                              a=a, b=b, c=c, d=d)


    # Plot
    plt.plot(t_e, N1_e, label=r'$N_1$ Euler', lw=2, color='blue')
    plt.plot(t_e, N2_e, label=r'$N_2$ Euler', lw=2, color='red')
    plt.plot(t_r, N1_r, linestyle=':', lw=3, color='blue', label=r'$N_1$ RK8')
    plt.plot(t_r, N2_r, linestyle=':', lw=3, color='red', label=r'$N_2$ RK8')
    plt.title("Lotka-Volterra Prey and Predator Model")
    plt.xlabel("Time (years)")
    plt.ylabel("Population/Carrying Cap.")
    plt.legend(loc='upper left', frameon=True)
    fig = plt.gcf()
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.text(0.98, 0.02, f"Coefficients: a={a}, b={b}, c={c}, d={d}",
            ha='right', va='bottom')

    plt.show()

    # Phase diagram
    plt.figure(figsize=(6, 5.2))
    # Trajectory（Euler & RK8）
    plt.plot(N1_e, N2_e, color='0.6', lw=2, label='Euler trajectory')
    plt.plot(N1_r, N2_r, 'b:', lw=2.5, label='RK8 trajectory')

    # Start and end point
    plt.plot(N1_e[0], N2_e[0], 'ko', ms=6, label='start')
    plt.plot(N1_e[-1], N2_e[-1], 'ks', ms=6, label='end')

    # Nullclines：dN1/dt=0 -> N2=a/b；dN2/dt=0 -> N1=c/d
    if b > 0:
        plt.axhline(y=a/b, color='red', ls='--', lw=1.5, label='dN1/dt=0 (N2=a/b)')
    if d > 0:
        plt.axvline(x=c/d, color='blue', ls='--', lw=1.5, label='dN2/dt=0 (N1=c/d)')

    plt.xlabel("Prey $N_1$")
    plt.ylabel("Predator $N_2$")
    plt.title("Phase diagram (Prey vs Predator)")
    plt.xlim(left=0); plt.ylim(bottom=0)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()

    if ics:
        plt.figure(figsize=(7.2, 5.2))
        # draw nullclines once
        if b > 0: plt.axhline(y=a/b, color='0.3', ls=':', lw=1)
        if d > 0: plt.axvline(x=c/d, color='0.3', ls=':', lw=1)

        for ic in ics:
            N10, N20 = ic
            tr, N1r, N2r = solve_rk8(dNdt_predprey, N10, N20, dT=dt, t_final=T,
                                     a=a, b=b, c=c, d=d)
            line, = plt.plot(N1r, N2r, lw=2, label=f"IC={ic} trajectory")
            # equilibrium marker in matching color
            eql_x, eql_y = (c/d if d>0 else np.nan), (a/b if b>0 else np.nan)
            plt.scatter([eql_x], [eql_y], s=40, color=line.get_color(), zorder=3,
                        label="equilibrium" if ic == ics[0] else None)

        plt.xlabel("Prey N1")
        plt.ylabel("Predator N2")
        plt.title("Phase diagram — closed orbits at different radii")
        plt.xlim(left=0); plt.ylim(bottom=0)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()