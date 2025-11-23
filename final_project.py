#!/usr/bin/env python3

'''
Snow Avalanche Model - Question 1: Build and Validate
Basic validation of stability calculator with simple test cases.

To run validation:
    validate_basic()
    plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # m/s²

# Core Function: Stability Calculator

def calculate_stability(theta, h, rho, c, phi, debug=False):
    '''
    Calculate Mohr-Coulomb stability factor.
    
    S = (c + rho*g*h*cos(theta)*tan(phi)) / (rho*g*h*sin(theta))
    
    If S < 1: avalanche triggers
    If S >= 1: stable
    
    Parameters
    ----------
    theta : float
        Slope angle (degrees)
    h : float
        Snow depth (m)
    rho : float
        Snow density (kg/m³)
    c : float
        Cohesion
    phi : float
        Internal friction angle (degrees)
    debug : bool
        Print calculation details
    
    Returns
    -------
    S : float
        Stability factor
    '''
    
    # Convert to radians
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    
    # Driving stress (pulling snow down)
    tau_drive = rho * g * h * np.sin(theta_rad)
    
    # Normal stress (pressing on slope)
    sigma_n = rho * g * h * np.cos(theta_rad)
    
    # Resisting strength (cohesion + friction)
    tau_resist = c + sigma_n * np.tan(phi_rad)
    
    # Stability factor
    S = tau_resist / tau_drive
    
    return S


# Question 1: Basic Validation

def validate_basic():
    '''
    Question 1: Build and validate the stability calculator.
    
    Test three simple cases:
    1. Gentle slope (should be stable)
    2. Moderate slope (borderline)
    3. Steep slope (should be unstable)
    
    This validates that:
    - The function runs without errors
    - S decreases as slope angle increases
    - S can be > 1 and < 1
    '''
    
    # Fixed snow properties (typical settled snow)
    h = 1.5      # 1.5 meters deep
    rho = 300    # 300 kg/m³ density
    c = 1000     # 1000 Pa cohesion
    phi = 28     # 28° friction angle
    
    # Test case 1
    theta1 = 25
    S1 = calculate_stability(theta1, h, rho, c, phi, debug=True)
    
    # Test case 2
    theta2 = 35
    S2 = calculate_stability(theta2, h, rho, c, phi, debug=True)
    
    # Test Case 3
    theta3 = 45
    S3 = calculate_stability(theta3, h, rho, c, phi, debug=True)
    
    # Check if validation passes
    passed = True
    if S1 <= S2 or S2 <= S3:
        print("PASS: S decreases as slope angle increases")
    else:
        print("FAIL: S should decrease with steeper slopes")
        passed = False
    
    if S1 > 1 and S3 < 1:
        print("PASS: Can identify both stable and unstable conditions")
    else:
        print("FAIL: Should find stable (gentle) and unstable (steep) cases")
        passed = False
    
    if passed:
        print("\ALL VALIDATION TESTS PASSED!")
    
    # Create visualization
    plot_validation(theta1, theta2, theta3, S1, S2, S3, h, rho, c, phi)
    
    return S1, S2, S3


def plot_validation(theta1, theta2, theta3, S1, S2, S3, h, rho, c, phi):
    '''
    Create simple validation plots for Question 1.
    Simple style matching the lab examples.
    '''
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Bar chart of three test cases
    test_names = ['Case 1\nGentle\n(25°)', 
                  'Case 2\nModerate\n(35°)', 
                  'Case 3\nSteep\n(45°)']
    S_values = [S1, S2, S3]
    colors = ['green' if s >= 1 else 'red' for s in S_values]
    
    axes[0].bar(test_names, S_values, color=colors, alpha=0.6)
    axes[0].axhline(y=1, color='black', linestyle='--', linewidth=2, 
                    label='Critical (S=1)')
    axes[0].set_ylabel('Stability Factor S')
    axes[0].set_title('Validation Test Results')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add S values on bars (simple version)
    for i, (name, s) in enumerate(zip(test_names, S_values)):
        axes[0].text(i, s + 0.1, f'S={s:.2f}', 
                    ha='center', fontsize=10)
    
    # Right: S vs slope angle curve
    theta_range = np.linspace(20, 50, 100)
    S_curve = [calculate_stability(th, h, rho, c, phi) for th in theta_range]
    
    axes[1].plot(theta_range, S_curve, 'b-', linewidth=2, label='S vs angle')
    axes[1].axhline(y=1, color='black', linestyle='--', linewidth=2, 
                    label='Critical (S=1)')
    
    # Mark three test points
    axes[1].plot([theta1, theta2, theta3], [S1, S2, S3], 
                 'ro', markersize=8, label='Test cases')
    
    axes[1].set_xlabel('Slope Angle (degrees)')
    axes[1].set_ylabel('Stability Factor S')
    axes[1].set_title('Stability vs Slope Angle')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()

