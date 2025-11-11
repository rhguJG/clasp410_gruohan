#!/usr/bin/env python3

'''
A module for burning forests and making pestilence.
What a happy coding time.
'''

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def forest_fire(isize=3, jsize=3, nstep=4, pspread=1.0, pignite=0.0, pbare=0.0, disease=False, pfatal=0.0):
    '''
    Create a forest fire.

    Parameters
    ----------
    isize, jsize : int, defaults to 3
        Set size of forest in x and y direction, respectively.
    nstep : int, defaults to 4
        Set number of steps to advance solution.
    pspread : float, defaults to 1.0
        Set chance that fire can spread in any direction, from 0 to 1
        (i.e., 0% to 100% chance of spread.)
    pignite : floate, defaults to 0.0
        Set the chance that a point starts the simulation on fire (or infected) from 0 to 1 (0% to 100%).
    pbare : floate, defaults to 0.0
        Set the chance that a point starts the simulation on bare (or immune) from 0 to 1 (0% to 100%)
    disease : bool
        If True, run disease mode (adds mortality vs immunity outcome).
    pfatal : float in [0,1], defaults to 0.0
        In disease mode, probability that a sick cell dies at the end of the step.
    '''

    # Creating a forest and making all spots have trees.
    forest = np.zeros((nstep, isize, jsize)) + 2

    # Set initial conditions for BURNING/INFECTED and BARE/IMMUNE
    # Start with bare land/immune people:
    loc_bare = rand(isize, jsize) <= pbare
    forest[0, loc_bare] = 1

    # Set up BURNING/INFECTED:
    if pignite > 0: # Scatter fire randomly:
        loc_ignite = np.zeros((isize, jsize), dtype=bool)
        while loc_ignite.sum() == 0:
            loc_ignite = rand(isize, jsize) <= pignite
        print(f"Starting with {loc_ignite.sum()} points on fire or infected.")
        # Only set to 3 if the cell is not already 1
        mask = loc_ignite & (forest[0] != 1)
        forest[0][mask] = 3
    else:
        # Set initial fire to center [NEED TO UPDATE THIS FOR LAB]:
        forest[0, isize//2, jsize//2] = 3

    # Loop through time to advance our fire.
    for k in range(nstep-1):
        # Assume the next time step is the same as the current:
        forest[k+1, :, :] = forest[k, :, :]
        # Search every spot that is on fire and spread fire as needed.
        for i in range(isize):
            for j in range(jsize):
                # Are we on fire?
                if forest[k, i, j] != 3:
                    continue
                # Ah! it burns. Spread fire in each direction.
                # Spread "up" (i to i-1)
                if (pspread > rand()) and (i > 0) and (forest[k, i-1, j] == 2):
                    forest[k+1, i-1, j] = 3
                # Spread "Down" (i to i+1)
                if (pspread > rand()) and (i < isize - 1) and (forest[k, i+1, j] == 2):
                    forest[k+1, i+1, j] = 3
                # Spread "East" (j to j+1)
                if (pspread > rand()) and (j < jsize - 1) and (forest[k, i, j+1] == 2):
                    forest[k+1, i, j+1] = 3
                # Spread "West" (j to j-1)
                if (pspread > rand()) and (j > 0) and (forest[k, i, j-1] == 2):
                    forest[k+1, i, j-1] = 3

                # Change buring to burnt:
                forest[k+1, i, j] = 1
                
                # This is for question 3. If we are considering the disease scenario, we will need to consider
                # the probability of fatality, and set status to 0.
                if disease:
                    # dies with probability pfatal, otherwise becomes immune
                    if (pfatal > rand()):
                        forest[k+1, i, j] = 0

    return forest

def plot_progression(forest):
    '''Calculate the time dynamics of a forest fire and plot them.'''

    # Get total number of points:
    ksize, isize, jsize = forest.shape
    npoints = isize * jsize

    # Find all spots that have forests (or are healthy people)
    # ...and count them as a function of time.
    loc_forest = forest == 2
    forested = 100 * loc_forest.sum(axis=(1,2)) / npoints

    loc_bare = forest == 1
    bare = 100 * loc_bare.sum(axis=(1,2)) / npoints

    plt.plot(forested, label='Forested')
    plt.plot(bare, label='Bare/Burnt')
    plt.legend()   
    plt.xlabel('Time (arbitrary units)')
    plt.ylabel('Percent Total Forest')

    # observables: t_stop (last step with positive burn rate) and B_final
    dB = np.diff(bare, prepend=bare[0])
    tol = 1e-9
    pos = np.where(dB > tol)[0]
    t_stop = int(pos[-1]) if pos.size > 0 else 0
    B_final = float(bare[t_stop])
    print(f"t_stop={t_stop}, B_final={B_final:.1f}%")

def plot_forest2d(forest, disease=False):
    '''Plot a 2D forest grid.
    Values: wildfire → 1=Bare, 2=Forest, 3=FIRE!
            disease  → 0=Dead, 1=Immune, 2=Healthy, 3=Sick'''
    # Choose a discrete colormap and the numeric range that maps to it.
    # Setting vmin/vmax fixes consistent colors across figures.
    if disease:
        forest_cmap = ListedColormap(['darkgrey', 'tan', 'forestgreen', 'crimson'])
        vmin, vmax = 0, 3
        label = {0:"Dead", 1:"Immune", 2:"Healthy", 3:"Sick"}
    else:
        forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
        vmin, vmax = 1, 3
        label = {1:"Bare", 2:"Forest", 3:"FIRE!"}

    ksize, isize, jsize = forest.shape
    A = np.asarray(forest)

    # One figure per time step (keeps the teacher’s pcolor style).
    for k in range(ksize):
        fig, ax = plt.subplots(1, 1)
        # Core heatmap: values in A[k] are mapped to the fixed color scale.
        ax.pcolor(A[k], cmap=forest_cmap, vmin=vmin, vmax=vmax)

        # Annotate each cell with a label and its coordinate.
        for i in range(isize):
            for j in range(jsize):
                v = int(A[k, i, j])
                ax.text(j+0.5, i+0.5, f"{label.get(v,'')}\n i, j = {i}, {j}",
                        ha='center', va='center', fontsize=9, color='black')

        # Put ticks on cell edges and lock limits to the grid size.
        ax.set_xticks(np.arange(0, jsize+1, 1))
        ax.set_yticks(np.arange(0, isize+1, 1))
        ax.set_xlim(0, jsize)
        ax.set_ylim(0, isize)
        # Square cells and clear white grid lines between them for readability.
        ax.set_aspect('equal')
        ax.grid(color='white', linewidth=2)
        # Axis labels and a descriptive title that reflects the mode and step.
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_title(f"{'Disease' if disease else 'Forest'} Status (iStep={k})")

def Q2_experiment1():
    isize=30; jsize=30; nstep=40; pignite=0.02; pbare=0.1
    ps_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(ps_list)))
    plt.figure()
    ax = plt.gca()
    for idx, ps in enumerate(ps_list):
        forest = forest_fire(isize=isize, jsize=jsize, nstep=nstep, pspread=ps, pignite=pignite, pbare=pbare)
        plot_progression(forest)
        line_forest = ax.lines[-2]
        line_bare = ax.lines[-1]
        line_forest.set_color(colors[idx])
        line_forest.set_linewidth(2)
        line_bare.set_color(colors[idx])  
        line_bare.set_linestyle('--')  
        line_bare.set_linewidth(2)
        line_forest.set_label(f"Forested (pspread={ps})")
        line_bare.set_label(f"Bare (pspread={ps})")
    plt.title(f"Progression (pignite={pignite}, pbare={pbare})",fontsize=16)
    plt.xlabel("Time (steps)"); plt.ylabel("Percent Total Forest")
    plt.legend(fontsize=9)
    plt.show()

def Q2_experiment2():
    isize=30; jsize=30; nstep=40; pspread=0.60; pignite=0.02
    pb_list = [0.0, 0.1, 0.3, 0.6, 0.8, 1.0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(pb_list)))

    plt.figure()
    ax = plt.gca()

    for idx, pb in enumerate(pb_list):
        forest = forest_fire(isize=isize, jsize=jsize, nstep=nstep, pspread=pspread, pignite=pignite, pbare=pb)
        plot_progression(forest)             # adds 2 lines: Forested then Bare/Burnt
        line_forest = ax.lines[-2]
        line_bare   = ax.lines[-1]

        # same color for the pair; Forested dashed, Bare dotted
        line_forest.set_color(colors[idx]);line_forest.set_linewidth(2)
        line_bare.set_color(colors[idx]);   line_bare.set_linestyle('--');    line_bare.set_linewidth(2)

        line_forest.set_label(f"Forested (pbare={pb})")
        line_bare.set_label(f"Bare (pbare={pb})")

    ax.set_title(f"Progression (pspread={pspread}, pignite={pignite})", fontsize=16)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Percent Total Forest")

    # legend outside (optional)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, borderaxespad=0.)
    plt.tight_layout(); plt.subplots_adjust(right=0.78)

    plt.show()

def Q3_experiment1():
    isize=30; jsize=30; nstep=50
    pspread=0.60
    pignite=0.02
    pbare=0.1 
    pf_list = [0.00, 0.20, 0.50, 0.80]
    colors = plt.cm.tab10(np.linspace(0, 1, len(pf_list))) 
    plt.figure()
    ax = plt.gca()
    for idx, pf in enumerate(pf_list):
        forest = forest_fire(isize=isize, jsize=jsize, nstep=nstep, pspread=pspread, pignite=pignite, pbare=pbare,
                            disease=True, pfatal=pf)
        plot_progression(forest)  # prints t_stop and B_final
        line_forest = ax.lines[-2]
        line_bare = ax.lines[-1]
        line_forest.set_color(colors[idx])
        line_forest.set_linewidth(2)
        line_bare.set_color(colors[idx])
        line_bare.set_linestyle('--')
        line_bare.set_linewidth(2)

        line_forest.set_label(f"Healthy (Psurvival={1-pf})")
        line_bare.set_label  (f"Immune-only (Psurvival={1-pf})")

    ax.set_title(f"Disease progression for different mortality(pspread={pspread}, pignite={pignite}, pbare={pbare})", fontsize=14)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Percent of population")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), fontsize=9, borderaxespad=0.)
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    plt.show()

def Q3_experiment2():
    isize=30; jsize=30; nstep=50
    pspread=0.60
    pignite=0.02
    pfatal=0.30         # Psurvive = 0.70 
    pb_list = [0.00, 0.10, 0.30, 0.60, 0.80]
    colors = plt.cm.tab10(np.linspace(0, 1, len(pb_list))) 
    plt.figure()
    ax = plt.gca()
    for idx, pb in enumerate(pb_list):
        forest = forest_fire(isize=isize, jsize=jsize, nstep=nstep, pspread=pspread, pignite=pignite, pbare=pb,
                            disease=True, pfatal=pfatal)
        plot_progression(forest)  # prints t_stop and B_final
        line_forest = ax.lines[-2]; line_bare = ax.lines[-1]
        line_forest.set_color(colors[idx]); line_forest.set_linewidth(2)
        line_bare.set_color(colors[idx]);   line_bare.set_linestyle('--'); line_bare.set_linewidth(2)

        line_forest.set_label(f"Healthy (pbare={pb})")
        line_bare.set_label  (f"Immune-only (pbare={pb})")

    ax.set_title(f"Disease progression for different early vaccine rates (pspread={pspread}, pignite={pignite}, pfatal={pfatal})", fontsize=14)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Percent of population")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), fontsize=9, borderaxespad=0.)
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    plt.show()