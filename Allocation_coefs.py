# -*- coding: utf-8 -*-
"""
Script created to demonstrate different resource allocation models (exponential, linear, hyperbolic) 
and their visualization. Suitable for applications needing resource distribution across groups or bands.

Author: presvotscor
Date Created: Thu Aug 8, 2024
"""

import numpy as np
import matplotlib.pyplot as plt

class Allocation:
    def __init__(self):
        """ Initialize any required variables (not used in this implementation). """
        pass

    def exp_model(self, Qmax, Qmin, M):
        """
        Calculates exponential decreasing values from Qmax to Qmin over M intervals.
        
        Args:
        - Qmax (float): Maximum value.
        - Qmin (float): Minimum value.
        - M (int): Number of intervals.
        
        Returns:
        - list: List of values.
        """
        return [Qmin * ((Qmax/Qmin) ** (-2**(m-M-1) + 1)) for m in range(1, M+2)]

    def linear_model(self, Qmax, Qmin, M):
        """
        Computes linearly decreasing values from Qmax to Qmin over M intervals.
        
        Args:
        - Qmax (float): Maximum value.
        - Qmin (float): Minimum value.
        - M (int): Number of intervals.
        
        Returns:
        - list: List of values.
        """
        return [(Qmax - Qmin) * (-2**(m - M - 1) + 1) + Qmin for m in range(1, M+2)]

    def hyperbolic_model(self, Qmax, Qmin, M, theta):
        """
        Calculates values based on a hyperbolic model from Qmax to Qmin over M intervals,
        influenced by the theta parameter.

        Args:
        - Qmax (float): Maximum value.
        - Qmin (float): Minimum value.
        - M (int): Number of intervals.
        - theta (float): Influences the shape of the hyperbolic curve.
        
        Returns:
        - list: List of values.
        """
        phi = 2 ** 6
        return [(Qmax - Qmin) * (1 - np.tanh(2 ** (-M - 1) * theta * (2 ** (m) - 0.5 * phi))) + Qmin for m in range(1, M+2)]

    def allocation_per_coeffs(self, N, M, Qmax, Qmin, model, theta, C):
        """
        Allocates resources according to the specified model across N units.

        Args:
        - N (int): Total number of units.
        - M (int): Number of bands.
        - Qmax, Qmin (float): Maximum and minimum values.
        - model (str): Type of model ('exp', 'linear', 'hyperbolic').
        - theta (float): Parameter for the hyperbolic model.
        - C (list): Coefficients dictating the number of units each band receives.

        Returns:
        - tuple: Tuple of raw and rounded allocation arrays.
        """
        if model == "exp":
            B = self.exp_model(Qmax, Qmin, M)
        elif model == "linear":
            B = self.linear_model(Qmax, Qmin, M)
        elif model == "hyperbolic":
            B = self.hyperbolic_model(Qmax, Qmin, M, theta)

        # Ensuring the allocations are within the bounds
        B = [np.max([np.min([b, Qmax]), Qmin]) for b in B]
        B_round = np.ceil(B)

        allocation = np.zeros(N, dtype=float)
        allocation_round = np.zeros(N, dtype=int)

        j = 0
        for m in range(M + 1):
            allocation[j:j + C[m]] = B[m]
            allocation_round[j:j + C[m]] = B_round[m]
            j += C[m]

        return allocation, allocation_round

if __name__ == "__main__":
    N = 1024
    M = int(np.log2(N))
    Qmax = 8
    Qmin = 2
    from DWT import Wavelet
    C = Wavelet().nb_coeffs_DWT_per_bande(N, M)

    Al = Allocation()
    theta1 = 2**1
    theta2 = 2**7

    # Generating allocations using different models
    allocation_exp, _ = Al.allocation_per_coeffs(N, M, Qmax, Qmin, "exp", None, C)
    allocation_linear, _ = Al.allocation_per_coeffs(N, M, Qmax, Qmin, "linear", None, C)
    allocation_hyperbolic1, _ = Al.allocation_per_coeffs(N, M, Qmax, Qmin, "hyperbolic", theta1, C)
    allocation_hyperbolic2, _ = Al.allocation_per_coeffs(N, M, Qmax, Qmin, "hyperbolic", theta2, C)

    # Plotting results
    plt.figure(figsize=(10, 4), dpi=100)
    plt.plot(range(N), allocation_exp, lw=2, label="Exponential")
    plt.plot(range(N), allocation_linear, lw=2, label='Linear')
    plt.plot(range(N), allocation_hyperbolic1, lw=2, label='Hyperbolic Theta1')
    plt.plot(range(N), allocation_hyperbolic2, lw=2, label='Hyperbolic Theta2')
    plt.xlabel('Samples')
    plt.ylabel('Allocated Quantity')
    plt.legend()
    plt.title(f'Allocation Models Comparison, Qmax={Qmax}, Qmin={Qmin}, M={M}')
    plt.grid(True)
    plt.minorticks_on()
    plt.show()
