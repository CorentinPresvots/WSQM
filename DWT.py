# -*- coding: utf-8 -*-
"""
Created on Thu Aug 8 12:35:55 2024

@author: presvotscor
"""

import pywt  # Import PyWavelets for wavelet transformations
import numpy as np  # Import NumPy for numerical operations

class Wavelet:
    """
    Class for handling Discrete Wavelet Transform (DWT) operations.
    """

    def __init__(self):
        # Set the wavelet mode to "periodization" for boundary handling
        self.mode = "periodization"

    def nb_coeffs_DWT_per_bande(self, N, level):
        """
        Computes the number of wavelet coefficients per frequency band for a given decomposition level.

        Parameters:
            N (int): Number of samples in the signal.
            level (int): Number of decomposition levels in the wavelet transform.

        Returns:
            Liste (np.array): An array containing the number of coefficients per frequency band.
        """
        Liste = np.zeros(level + 1, dtype=int)
        for k in range(0, level + 1):
            Liste[level + 1 - k - 1] = N // 2**(k + 1)  # Compute coefficients per level
        Liste[0] = Liste[1]  # Ensure consistency for the first level
        return Liste

    def get_coefs_DWT(self, x, wave, level):
        """
        Computes the Discrete Wavelet Transform (DWT) coefficients of a given signal.

        Parameters:
            x (np.array): Input signal.
            wave (str): Wavelet type (e.g., "coif5").
            level (int): Number of decomposition levels.

        Returns:
            coefs_DWT (np.array): Array containing all wavelet coefficients.
        """
        coefs = pywt.wavedec(x, wave, mode=self.mode, level=level)  # Perform wavelet decomposition
        coefs_DWT = []
        
        # Flatten the coefficient list into a single array
        for i in range(level + 1):
            coefs_DWT.extend(coefs[i])

        return np.array(coefs_DWT)

    def get_x_rec_DWT(self, coefs_DWT, wave, level):
        """
        Reconstructs the signal from its wavelet coefficients.

        Parameters:
            coefs_DWT (np.array): Wavelet coefficients.
            wave (str): Wavelet type used for decomposition.
            level (int): Number of decomposition levels.

        Returns:
            x_rec (np.array): Reconstructed signal.
        """
        # Retrieve the number of coefficients per frequency band
        nb_coeffs_per_bande = self.nb_coeffs_DWT_per_bande(len(coefs_DWT), level)
        
        coefs_dec = []
        ptr = 0
        for i in range(level + 1):
            coefs_dec.append(np.array(coefs_DWT[ptr:ptr + nb_coeffs_per_bande[i]]))
            ptr += nb_coeffs_per_bande[i]

        # Perform inverse wavelet transform to reconstruct the signal
        x_rec = pywt.waverec(coefs_dec, wave, mode=self.mode)

        return x_rec


# Main program
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Normalize import normalize  # Import a normalization function from an external module
    from Measures import get_snr  # Import an SNR computation function

   

    # Import test signal generation function
    from get_test_signal import get_RTE_signal as get_signal
    
    # Define parameters
    fs = 6400  # Sampling frequency in Hz
    fn = 50    # Nominal signal frequency in Hz
    N = 128   # Number of samples

    # Define processing parameters
    nb_signal = 5  # Number of signals
    nb_phase = 3  # Number of phases per signal
    nb_w = 40  # Number of time windows

    wave = "coif5"  # Chosen wavelet function
    level = int(np.log2(N))  # Compute the wavelet decomposition level

    L_coefs = []  # List to store wavelet coefficients
    L_SNR = []  # List to store computed SNR values

    # Initialize wavelet processing class
    W = Wavelet()

    # Process signals
    for id_signal in range(nb_signal):
        for id_phase in range(nb_phase):
            for w in range(20, nb_w):  # Process only selected windows
                x = get_signal(id_signal)[id_phase][w*N:(w+1)*N]  # Extract signal window
                
                # Compute wavelet coefficients
                coefs_DWT = W.get_coefs_DWT(x, wave, level)
                
                # Normalize coefficients
                coefs_DWT_n, kx = normalize(coefs_DWT)

                # Store coefficients
                L_coefs.append(coefs_DWT_n)

                # Reconstruct signal from wavelet coefficients
                x_rec = W.get_x_rec_DWT(coefs_DWT, wave, level)

                # Compute and store SNR
                L_SNR.append(get_snr(x, x_rec))

    # Plot mean magnitude of wavelet coefficients
    plt.figure(figsize=(8, 4), dpi=80)
    plt.plot(np.mean(np.abs(L_coefs), axis=0), lw=2, label='Mean coefficients')
    plt.xlabel('Index')
    plt.ylabel("Magnitude")
    plt.title('Mean Magnitude of Coefficients over {} Windows, SNR Mean = {:.1f} dB'.format(len(L_SNR), np.mean(L_SNR)))
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)

    plt.show()
