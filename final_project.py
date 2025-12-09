#!/usr/bin/env python3

'''
A module for simulating avalanche dynamics and snow stability.
For Q1 validation, please run:
sim, hcrit = question_1()
plt.show()

For Q2 curved slope analysis, please run:
A, B, C = question_2()
plt.show()

For Q3 flow rate comparison, please run:
sim_a, sim_b = question_3()
plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Physical constants
RHO_SNOW = 300.0 # snow density (kg/m^3)
G = 9.81 # gravitational acceleration (m/s^2)


def get_stability(h, theta_deg, C, phi_deg, rho=RHO_SNOW, g=G):
    '''
    Compute Mohr-Coulomb stability ratio.

    Parameters
    ----------
    h : array-like
        Snow depth at each location.
    theta_deg : float or array-like
        Slope angle in degrees.
    C : float or array-like
        Cohesion (Pa).
    phi_deg : float or array-like
        Friction angle in degrees.
    rho : float, defaults to RHO_SNOW
        Snow density (kg/m^3).
    g : float, defaults to G
        Gravitational acceleration (m/s^2).

    Returns
    -------
    S : ndarray
        Stability ratio. Values < 1.0 indicate failure.
        S = (C + rho g h cos(theta) tan(phi)) / (rho g h sin(theta))
    '''
    # Convert angles from degrees to radians
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    # Downslope (shear) and normal components of weight
    shear = rho * g * h * np.sin(theta)
    normal = rho * g * h * np.cos(theta)

    # Mohr–Coulomb shear strength
    strength = C + normal * np.tan(phi)

    # Allocate output array and safely divide
    S = np.empty_like(strength, dtype=float)
    np.divide(strength, shear, out=S, where=(shear > 0))
    # If there is no shear (e.g. zero depth), treat as perfectly stable
    S[~(shear > 0)] = np.inf
    return S


def compute_critical_depth(theta_deg, C, phi_deg, rho=RHO_SNOW, g=G):
    '''
    Compute analytical critical depth from Mohr-Coulomb theory.

    Parameters
    ----------
    theta_deg : float
        Slope angle in degrees.
    C : float
        Cohesion (Pa).
    phi_deg : float
        Friction angle in degrees.
    rho : float, defaults to RHO_SNOW
        Snow density (kg/m^3).
    g : float, defaults to G
        Gravitational acceleration (m/s^2).

    Returns
    -------
    h_crit : float
        Critical depth from S = 1:
        h_crit = C / [rho g (sin(theta) - cos(theta) tan(phi))]
    '''
    # Convert to radians once
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    # Denominator of analytic expression
    denom = rho * g * (np.sin(theta) - np.cos(theta) * np.tan(phi))
    if denom <= 0:
        # If denominator is non-positive, slope is never unstable
        return np.inf
    return C / denom


def simulate(x, theta_deg, Sx, Cx, *,
             phi_deg=35.0, dt=0.1, steps=600, flow_rate=0.30,
             t_snow_hours=np.inf, h0=None):
    '''
    Simulate snowpack evolution with cascaded transport.

    Parameters
    ----------
    x : array-like
        Spatial grid positions (m).
    theta_deg : float or array-like
        Slope angle(s) in degrees.
    Sx : float or array-like
        Snowfall rate(s) (m/hr).
    Cx : float or array-like
        Cohesion value(s) (Pa).
    phi_deg : float, defaults to 35.0
        Friction angle in degrees.
    dt : float, defaults to 0.1
        Time step (hours).
    steps : int, defaults to 600
        Number of time steps.
    flow_rate : float, defaults to 0.30
        Flow rate parameter controlling avalanche transport (0 to 1).
    t_snow_hours : float, defaults to np.inf
        Duration of snowfall (hours).
    h0 : array-like, optional
        Initial snow depth. Defaults to zeros.

    Returns
    -------
    dict
        Dictionary containing:
        - h_history: depth evolution (steps+1, n)
        - S_history: stability evolution (steps+1, n)
        - time: time array
        - x: spatial grid
        - theta: slope angles
        - outflow: total mass that exited domain
        - first_failure_time: when S < 1 first occurs (hours)
        - first_failure_loc: where first failure occurs (m)
    '''
    n = len(x)

    # Broadcast scalar inputs to full-length arrays if needed
    theta = np.atleast_1d(theta_deg) * np.ones(n)
    Sx = np.atleast_1d(Sx) * np.ones(n)
    Cx = np.atleast_1d(Cx) * np.ones(n)

    # Initial snow depth (zeros or provided profile)
    h = np.zeros(n, float) if h0 is None else np.asarray(h0, float).copy()

    # Allocate history arrays
    time = np.arange(steps + 1) * dt
    h_hist = np.zeros((steps + 1, n), float)
    S_hist = np.zeros((steps + 1, n), float)

    # Store initial state
    h_hist[0] = h.copy()
    S_hist[0] = get_stability(h, theta, Cx, phi_deg)

    outflow = 0.0
    first_failure_time = None
    first_failure_loc = None

    # Tune cascaded transport coefficients from flow_rate parameter
    # alpha, gamma control how much snow continues to propagate downslope
    alpha = float(np.clip(0.25 + 0.80 * flow_rate, 0.0, 0.95))
    gamma = float(np.clip(0.15 + 0.70 * flow_rate, 0.0, 0.90))

    # Main time-stepping loop
    for k in range(steps):
        t_now = (k + 1) * dt

        # Add snowfall up to t_snow_hours
        if t_now <= t_snow_hours:
            h += Sx * dt

        # Check stability before transport (for failure detection)
        S_pre = get_stability(h, theta, Cx, phi_deg)
        if first_failure_time is None and np.any(S_pre < 1.0):
            # Record the first time and location where S < 1
            first_failure_time = t_now
            first_failure_loc = x[np.argmin(S_pre)]

        # Cascaded transport (top to bottom sweep)
        # Ratio = S_pre encodes local stability at current step
        ratio = S_pre
        for i in range(n - 1, -1, -1):
            # Skip stable or empty cells
            if ratio[i] >= 1.0 or h[i] <= 0.0:
                continue

            # Start with a fraction of snow that becomes mobile
            carry = h[i] * flow_rate
            h[i] -= carry

            # Move snow one cell downslope at a time
            j = i - 1
            while carry > 1e-9 and j >= 0:
                # Deposit current carry at location j
                h[j] += carry

                # Re-check receiver stability using local 1-point computation
                rj = float(get_stability(np.array([h[j]]),
                                         np.array([theta[j]]),
                                         np.array([Cx[j]]),
                                         phi_deg))

                # If still unstable, keep pushing snow downslope (spill)
                if rj < 1.0:
                    spill = alpha * flow_rate * h[j]
                else:
                    # Otherwise, snow spreads but with weaker spill
                    spill = gamma * h[j]

                # Update local depth and new carry
                h[j] -= spill
                carry = spill
                j -= 1

            # If we exit the domain (j < 0), record outflow
            if carry > 0.0 and j < 0:
                outflow += carry

        # Record this timestep
        h_hist[k + 1] = h.copy()
        S_hist[k + 1] = get_stability(h, theta, Cx, phi_deg)

    return dict(
        h_history=h_hist,
        S_history=S_hist,
        time=time,
        x=x,
        theta=theta,
        outflow=outflow,
        first_failure_time=first_failure_time,
        first_failure_loc=first_failure_loc,
    )

# Unified metrics
TAU_FILM = 0.03  # Thin-film cutoff threshold (m)


def _thin_film(h, tau=TAU_FILM):
    '''Remove depths below thin-film threshold.'''
    # Everything thinner than tau is treated as numerical film and zeroed
    return np.where(h < tau, 0.0, h)


def _retained_mass(h, x):
    '''Calculate total retained mass via ∫ h dx.'''
    # Simple trapezoidal integration of depth profile
    return float(np.trapz(h, x))


def _centroid_x(h, x):
    '''Calculate mass-weighted centroid position.'''
    A = _retained_mass(h, x)
    # Avoid division by zero for empty domain
    return (float(np.trapz(x * h, x)) / A) if A > 0 else np.nan


def _runout_front_x(h, x, tau=TAU_FILM):
    '''Find lowest position with h(x) > tau.'''
    idx = np.where(h > tau)[0]
    # By convention, take the lowest x index that still has snow above tau
    return float(x[idx[0]]) if idx.size else np.nan


def _occupied_extent_and_Lstar(h, x, tau=TAU_FILM):
    '''Calculate occupied extent and dimensionless runout.'''
    idx = np.where(h > tau)[0]
    if idx.size:
        # Physical length occupied by snow above tau
        L_occ = float(x[idx.max()] - x[idx.min()])
    else:
        L_occ = 0.0
    L = float(x[-1] - x[0])
    # Lstar is extent normalized by total slope length
    Lstar = (L_occ / L) if L > 0 else np.nan
    return L_occ, Lstar


def _peak_depth_and_x(h, x):
    '''Find maximum depth and its location.'''
    j = int(np.argmax(h))
    return float(h[j]), float(x[j])


def unified_metrics_from_sim(sim, x, *, tau=TAU_FILM, extra=None):
    '''
    Extract unified metrics from simulation.

    Parameters
    ----------
    sim : dict
        Simulation output from simulate().
    x : array-like
        Spatial grid.
    tau : float, defaults to TAU_FILM
        Thin-film threshold (m).
    extra : dict, optional
        Additional metrics to include.

    Returns
    -------
    dict
        Metrics including:
        - M: retained mass (m²)
        - xbar: mass centroid (m)
        - x_front: runout front position (m)
        - L_occ: occupied extent (m)
        - Lstar: dimensionless runout
        - h_peak: maximum depth (m)
        - x_peak: location of peak (m)
        - outflow: exit flux
        - tau: threshold used
    '''
    # Take final time step and enforce thin-film cutoff
    hfin = _thin_film(np.maximum(sim["h_history"][-1], 0.0), tau=tau)

    # Bulk-integral metrics
    M = _retained_mass(hfin, x)
    xbar = _centroid_x(hfin, x)
    x_front = _runout_front_x(hfin, x, tau=tau)
    L_occ, Lstar = _occupied_extent_and_Lstar(hfin, x, tau=tau)
    hpk, xpk = _peak_depth_and_x(hfin, x)

    # Amount of snow that left the domain during the run
    outflow = float(sim.get("outflow", float("nan")))

    out = dict(M=M, xbar=xbar, x_front=x_front, L_occ=L_occ, Lstar=Lstar,
               h_peak=hpk, x_peak=xpk, outflow=outflow, tau=tau)
    # Allow caller to append extra diagnostic metrics
    if isinstance(extra, dict):
        out.update(extra)
    return out


# Question 1
def question_1():
    '''
    Validate critical depth against analytical solution.

    Configuration
    ------
    - Uniform 40° slope.
    - Constant snowfall and cohesion.
    - Single failure point triggers avalanche.

    Returns
    -------
    sim : dict
        Simulation output.
    hcrit : float
        Theoretical critical depth (m).
    '''
    npoints = 100
    x = np.linspace(0, 1000, npoints)
    theta = np.full_like(x, 40.0)
    Sx = 0.10
    Cx = 500.0
    phi = 35.0

    # Theoretical critical depth from Mohr–Coulomb
    hcrit = compute_critical_depth(40.0, Cx, phi)

    # Run simulation with same parameters
    sim = simulate(x, theta, Sx, Cx, phi_deg=phi, dt=0.1, steps=600, flow_rate=0.30)

    # Compare modeled vs theoretical
    failure_mask = sim["S_history"] < 1.0
    if np.any(failure_mask):
        # first time index where any cell fails
        k_fail = np.argmax(np.any(failure_mask, axis=1))
        # average depth at that time step
        h_model = float(np.mean(sim["h_history"][k_fail]))
        err_pct = abs(h_model - hcrit) / hcrit * 100.0
    else:
        # No failure detected: treat as full mismatch
        k_fail, h_model, err_pct = -1, 0.0, 100.0

    # Plot depth evolution
    t = sim["time"]
    h_max = np.max(sim["h_history"], axis=1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(t, h_max, lw=2, label="Max depth")
    ax.axhline(hcrit, ls="--", c="crimson", lw=2, label=f"h_crit = {hcrit:.3f} m")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Depth Evolution and theoretical h_crit")
    ax.legend()
    ax.grid(alpha=0.3)

    print(f"[Q1] h_crit={hcrit:.4f} m,  modeled={h_model:.4f} m,  error={err_pct:.2f}%")
    return sim, hcrit


# Question 2
def question_2():
    '''
    Compare three scenarios on a curved slope (10° to 50°).

    Scenarios
    ---------
    A) Baseline: uniform snowfall and cohesion.
    B) Top-heavy snowfall: higher accumulation at summit.
    C) Weak summit: reduced cohesion near the top.

    All use unified metrics for comparison.

    Returns
    -------
    A, B, C : dict
        Three simulation outputs.
    '''
    # Common setup
    npoints = 100
    x = np.linspace(0, 1000, npoints)
    # Slope angle increases from 10° at bottom to 50° at top
    theta  = np.interp(x, [0, 1000], [10, 50])
    phi = 35.0

    # Scenario A: baseline (uniform snowfall and cohesion)
    S_A = 0.05 * np.ones(npoints)
    C_A = 500.0 * np.ones(npoints)

    # Scenario B: top-heavy snowfall (same mean as A)
    # More accumulation near the summit, less at the bottom
    S_B = np.interp(x, [0, 1000], [0.02, 0.08])
    C_B = 500.0 * np.ones(npoints)

    # Scenario C: weak summit cohesion
    # Cohesion decreases from 800 Pa at bottom to 200 Pa at top
    S_C = 0.05 * np.ones(npoints)
    C_C = np.interp(x, [0, 1000], [800, 200])

    # Run all three simulations
    A = simulate(x, theta, S_A, C_A, phi_deg=phi, dt=0.1, steps=600, flow_rate=0.30)
    B = simulate(x, theta, S_B, C_B, phi_deg=phi, dt=0.1, steps=600, flow_rate=0.30)
    C = simulate(x, theta, S_C, C_C, phi_deg=phi, dt=0.1, steps=600, flow_rate=0.30)

    # Final depth profiles
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(x, A["h_history"][-1], "k-", lw=2, label="A: baseline")
    ax1.plot(x, B["h_history"][-1], "b-", lw=2, label="B: top-heavy snowfall")
    ax1.plot(x, C["h_history"][-1], "r-", lw=2, label="C: weak summit cohesion")
    ax1.set_xlabel("Position (m)")
    ax1.set_ylabel("Final depth (m)")
    ax1.set_title("Final Snow Distribution")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Summit evolution
    summit = -1  # Last grid point = summit
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    ax2.plot(A["time"], A["h_history"][:, summit], "k-", lw=2, label="A: baseline")
    ax2.plot(B["time"], B["h_history"][:, summit], "b-", lw=2, label="B: top-heavy snowfall")
    ax2.plot(C["time"], C["h_history"][:, summit], "r-", lw=2, label="C: weak summit cohesion")
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Depth at summit (m)")
    ax2.set_title("Summit Depth Evolution")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Compute unified metrics
    SA = unified_metrics_from_sim(A, x, extra={"meanS": float(np.mean(S_A))})
    SB = unified_metrics_from_sim(B, x, extra={"meanS": float(np.mean(S_B))})
    SC = unified_metrics_from_sim(C, x, extra={"meanS": float(np.mean(S_C))})

    # Print a compact table of diagnostics across scenarios
    print(f"\n[Q2] Domain-mean snowfall (m/hr):    A={SA['meanS']:.5f}  B={SB['meanS']:.5f}  C={SC['meanS']:.5f}")
    print(f"[Q2] First trigger time (hr):        A={A['first_failure_time']:.1f}   "
          f"B={B['first_failure_time']:.1f}   C={C['first_failure_time']:.1f}")
    print(f"[Q2] Retained mass ∫h dx (m²):       A={SA['M']:.2f}   B={SB['M']:.2f}   C={SC['M']:.2f}")
    print(f"[Q2] Mass centroid x (m):            A={SA['xbar']:.1f}   B={SB['xbar']:.1f}   C={SC['xbar']:.1f}")
    print(f"[Q2] Runout front x@τ={SA['tau']:.2f} m:        A={SA['x_front']:.1f}   B={SB['x_front']:.1f}   C={SC['x_front']:.1f}")
    print(f"[Q2] L_occ (m) / L*:                 "
          f"A={SA['L_occ']:.1f}/{SA['Lstar']:.3f}   B={SB['L_occ']:.1f}/{SB['Lstar']:.3f}   C={SC['L_occ']:.1f}/{SC['Lstar']:.3f}")
    print(f"[Q2] Peak depth (m) @ x (m):         "
          f"A={SA['h_peak']:.2f}@{SA['x_peak']:.0f}   B={SB['h_peak']:.2f}@{SB['x_peak']:.0f}   C={SC['h_peak']:.2f}@{SC['x_peak']:.0f}")
    print(f"[Q2] Outflow (model units):          A={SA['outflow']:.3f}   B={SB['outflow']:.3f}   C={SC['outflow']:.3f}")

    return A, B, C


# Question 3
def question_3(phi_deg=25.0, t_snow_hours=20.0, dt=0.1, steps=600,
                         eps_film=0.03):
    '''
    Compare two flow rates on curved slope (same environment as Q2).

    Settings
    --------
    - flow_rate = 0.30: sticky snow (limited transport).
    - flow_rate = 0.50: fluid snow (enhanced transport).
    - Both use identical unified metrics for comparison.

    Parameters
    ----------
    phi_deg : float, defaults to 25.0
        Friction angle (degrees).
    t_snow_hours : float, defaults to 20.0
        Duration of snowfall (hours).
    dt : float, defaults to 0.1
        Time step (hours).
    steps : int, defaults to 600
        Number of time steps.
    eps_film : float, defaults to 0.03
        Film threshold for visualization (m).

    Returns
    -------
    sim_a, sim_b : dict
        Simulations for flow_rate=0.30 and 0.50.
    '''
    npoints = 120
    x = np.linspace(0, 1000, npoints)
    # Same curved slope as Question 2
    theta = np.interp(x, [0, 1000], [10, 50])
    # Top-heavy snowfall pattern
    Sx = np.interp(x, [0, 1000], [0.02, 0.08])
    Cx = np.full_like(x, 500.0)

    # Run two simulations with different flow rates
    sim_a = simulate(x, theta, Sx, Cx, phi_deg=phi_deg, dt=dt, steps=steps,
                     flow_rate=0.30, t_snow_hours=t_snow_hours)
    sim_b = simulate(x, theta, Sx, Cx, phi_deg=phi_deg, dt=dt, steps=steps,
                     flow_rate=0.50, t_snow_hours=t_snow_hours)

    # Prepare final depths for plotting (apply thin film for clarity)
    h_a = np.maximum(sim_a["h_history"][-1], 0.0)
    h_a[h_a < eps_film] = 0.0
    h_b = np.maximum(sim_b["h_history"][-1], 0.0)
    h_b[h_b < eps_film] = 0.0

    # Plots: stacked panels comparing sticky vs fluid snow
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.fill_between(x, h_a, color="#8B4513", alpha=0.6)
    ax1.plot(x, h_a, color="#8B4513", lw=2)
    ax1.set_title("Sticky Snow (flow = 0.30)")
    ax1.set_ylabel("Depth (m)")
    ax1.grid(alpha=0.3)

    ax2.fill_between(x, h_b, color="#1E90FF", alpha=0.6)
    ax2.plot(x, h_b, color="#1E90FF", lw=2)
    ax2.set_title("Fluid Snow (flow = 0.50)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_xlabel("Position (m)  [0 = Bottom, 1000 = Top]")
    ax2.grid(alpha=0.3)

    # Compute unified metrics
    print(f"\nQ3 quantitative summary (tau = {TAU_FILM:.2f} m):")
    m30 = unified_metrics_from_sim(sim_a, x)
    m50 = unified_metrics_from_sim(sim_b, x)

    # Print side-by-side diagnostics for the two flow rates
    print(f"[Q3] L*:                          flow=0.30: {m30['Lstar']:.3f}   flow=0.50: {m50['Lstar']:.3f}")
    print(f"[Q3] Retained mass ∫h dx (m²):    flow=0.30: {m30['M']:.2f}   flow=0.50: {m50['M']:.2f}")
    print(f"[Q3] Outflow (model units):       flow=0.30: {m30['outflow']:.2f}   flow=0.50: {m50['outflow']:.2f}")
    print(f"[Q3] Mass centroid x (m):         flow=0.30: {m30['xbar']:.1f}   flow=0.50: {m50['xbar']:.1f}")
    print(f"[Q3] Peak depth (m) @ x (m):      flow=0.30: {m30['h_peak']:.2f}@{m30['x_peak']:.0f}   flow=0.50: {m50['h_peak']:.2f}@{m50['x_peak']:.0f}")
    print(f"[Q3] Runout front x (m):          flow=0.30: {m30['x_front']:.1f}   flow=0.50: {m50['x_front']:.1f}")

    return sim_a, sim_b
