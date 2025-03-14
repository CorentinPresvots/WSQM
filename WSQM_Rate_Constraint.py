# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 19:36:30 2024

@author: presvotscor
"""

#%%  load database
import numpy as np
import matplotlib.pyplot as plt
import time
from Normalize import normalize
from Measures import get_snr,get_rmse
from Quantization import Quantizer

from Allocation_coefs import Allocation 
from RLE_Golomb_Arithmetic import Golomb,Golomb_encoding,Golomb_decoding,RLE_Golomb_encoding,RLE_Golomb_decoding
from DWT import Wavelet

# Import test signal generation function
from get_test_signal import get_RTE_signal as get_signal

# General parameters
fs = 6400  # Sampling frequency in Hz
fn = 50    # Nominal signal frequency in Hz
N = 128    # Number of samples per window

# Input parameters
nb_signal = 2     # Number of signals to be encoded
nb_phase = 1       # Number of signal phases (1 = single phase, 3 = voltage phases, 6 = voltages + currents)
nb_w = 50         # Number of non-overlapping windows per signal

n_tot=128


# Main program
adaptive = True  # Enable adaptive encoding

# Instantiate different components used for compression
Al = Allocation()  # Bit allocation object
G = Golomb()  # Golomb encoding object
G_enc = Golomb_encoding(adaptive)  # Golomb encoder
G_dec = Golomb_decoding(adaptive)  # Golomb decoder
RLE_G_enc = RLE_Golomb_encoding(adaptive)  # Run-Length Golomb encoder
RLE_G_dec = RLE_Golomb_decoding(adaptive)  # Run-Length Golomb decoder
W = Wavelet()  # Wavelet transform object
Q = Quantizer()  # Quantizer object

# Define wavelet type and decomposition level
wave = "coif5"
level = int(np.log2(N))  # Compute decomposition level based on signal size

# Initialize storage matrices for performance metrics
SNR = np.zeros((nb_signal, nb_phase, nb_w))  # Signal-to-noise ratio
RMSE = np.zeros((nb_signal, nb_phase, nb_w))  # Root Mean Square Error
R_flow = np.zeros((nb_signal, nb_phase, nb_w))  # Bit rate for encoding
R_h = np.zeros((nb_signal, nb_phase, nb_w))  # Header rate
R_unused = np.zeros((nb_signal, nb_phase, nb_w))  # Header rate
x_real = np.zeros((nb_signal, nb_phase, N * nb_w))  # Original signals
x_rec = np.zeros((nb_signal, nb_phase, N * nb_w))  # Reconstructed signals
M = np.zeros((nb_signal, nb_phase, nb_w))  # Selected models

# Define the set of models to be tested
Models = {}
id_model = 0
Models[id_model] = {"family": "exp", "name": "exp"}

id_model += 1
Models[id_model] = {"family": "linear", "name": "linear"}

# Uncomment to add hyperbolic models with different thresholds
"""
for i in range(1, 7,2):
    id_model += 1
    Models[id_model] = {"family": "hyperbolic", "name": f"hyp. th={2**i}", "theta": 2**i}
"""


# The variable 'nb_fact' is a scaling factor for quantization used to test a wider range of increasingly fine-grained non-integer 'Qmax' values.
nb_fact = 2

# Maximum quantization value for Qmax (scaled by nb_fact)
Qmax_max = 16 * nb_fact #For hight bit rate n_tot increase this number to not saturate the quantisation step

# Number of coefficients to encode directly without further processing
coefs_to_encode_directly = int(4 * N / (fs / fn))

# Compute the number of wavelet coefficients per band
nb_coeffs_DWT_per_bande = W.nb_coeffs_DWT_per_bande(N, level)

# Parameters for dictionary-based entropy encoding
m_values = 3  # Number of distinct values in the dictionary
m_counts = 6  # Number of counts for RLE encoding
size_dico = N  # Dictionary size based on signal length
n_value_max = 6  # Maximum number of values for encoding
pos_E = 3  # Position encoding parameter

# Create dictionaries for RLE-Golomb encoding
dico_counts = RLE_G_enc.creat_dico(m_counts, size_dico, None)  # Dictionary for counts
dico_values = RLE_G_enc.creat_dico(m_values, 2 ** n_value_max + 1, pos_E)  # Dictionary for values

# Compute the number of bits needed to encode Qmax and model selection
nb_Qmax = np.max([1, int(np.ceil(np.log2(Qmax_max)))])  # Bits required for Qmax representation
nb_m = np.max([1, int(np.ceil(np.log2(len(Models))))])  # Bits required for model selection
nb_kx = 5  # Fixed number of bits for additional model parameters

# Temporal segmentation: splitting the signal into bands of size N/dec
# Only non-zero coefficient bands are transmitted
dec = N  

# Header initialization: Bits needed to transmit the number of models (kx), Qmin, and Qmax
R_h += nb_kx + nb_m + 2 * nb_Qmax  

# Adjust header size based on whether decimation is used
if dec is None:
    R_h += level + 1  # Add level bits if no decimation is used
else:
    if N != dec:
        R_h += int(N / dec)  # Add additional bits for decimation if applicable
        
        

#Record the start time
start_time = time.time()

for id_signal in range(nb_signal):
    
    # Retrieve signal components (voltage and current phases)
    v1 = get_signal(id_signal)[0]
    v2 = get_signal(id_signal)[1]
    v3 = get_signal(id_signal)[2]
    i1 = get_signal(id_signal)[3]
    i2 = get_signal(id_signal)[4]
    i3 = get_signal(id_signal)[5]

    # Create input array for MMC (first N*nb_w samples per phase)
    x = [
        v1[:N * nb_w], v2[:N * nb_w], v3[:N * nb_w],
        i1[:N * nb_w], i2[:N * nb_w], i3[:N * nb_w]
    ]

                
    # Iterate over each phase of the signal
    for id_phase in range(nb_phase):  
    
        # Store the original signal for this phase
        x_real[id_signal][id_phase] = x[id_phase]
        
        # Process each window within the current phase
        for w in range(nb_w):
    
            # Extract the current window of samples
            x_test = x[id_phase][w * N:(w + 1) * N]
    
            # Compute the Discrete Wavelet Transform (DWT) coefficients
            coefs_DWT = W.get_coefs_DWT(x_test, wave, level)
            
            # Normalize the DWT coefficients
            coefs_DWT_n, k_DWT = normalize(coefs_DWT)
            
            # Initialize minimum RMSE to a very high value
            RMSE_min = np.infty
            
            # Iterate over all model candidates
            
            Qmax_test=2 # Initialize Qmax to 1 for the first model and use the optimal Qmax for subsequent models.
            for id_model in Models:
                end_Na = 0
                RMSE_model = np.infty
    
                # Iterate over possible quantization upper bounds (Qmax)
                for Qmax in range(Qmax_test, Qmax_max, 1):
                    Qmax_test = Qmax  # Store initial Qmax value
                    
                    # Iterate over possible quantization lower bounds (Qmin)
                    for Qmin in range(Qmax_test,0,- 1):#:#:(1,Qmax_test,1):#
                        Qmax = Qmax_test / nb_fact
                        Qmin /= nb_fact
    
                        # Allocate bits per band based on the selected model
                        if Models[id_model]["family"] == "hyperbolic":
                            theta = Models[id_model]["theta"]
                        else:
                            theta = None
    
                        allocation, allocation_round = Al.allocation_per_coeffs(
                            N, level, Qmax, Qmin, Models[id_model]["family"], theta, nb_coeffs_DWT_per_bande
                        )
    
                        # Perform quantization
                        ind_coefs_DWT_q_enc = np.zeros(N)
                        #coefs_DWT_q_enc = np.zeros(N)
    
                        for k in range(N):
                            dynamic = 2 ** (allocation[k] - allocation[0] + 1)
                            ind_coefs_DWT_q_enc[k] = Q.get_ind(coefs_DWT_n[k], allocation_round[k], dynamic, 0)
                            #coefs_DWT_q_enc[k] = Q.get_q(ind_coefs_DWT_q_enc[k], allocation[k], dynamic, 0)
    
                        # Remove zero-value bands to optimize compression
                        if dec is None:
                            ind_to_encode = []
                            label_to_encode = np.zeros(level + 1)
                            coefs_to_encode = np.zeros(N)
                            ptr = 0
                            for i in range(level + 1):
                                if np.mean(np.abs(ind_coefs_DWT_q_enc[ptr:ptr + nb_coeffs_DWT_per_bande[i]])) != 0:
                                    ind_to_encode.extend(ind_coefs_DWT_q_enc[ptr:ptr + nb_coeffs_DWT_per_bande[i]])
                                    label_to_encode[i] = 1
                                    coefs_to_encode[ptr:ptr + nb_coeffs_DWT_per_bande[i]] = 1
                                ptr += nb_coeffs_DWT_per_bande[i]
                        else:
                            ind_to_encode = []
                            label_to_encode = np.zeros(int(N / dec))
                            coefs_to_encode = np.zeros(N)
                            ptr = 0
                            for i in range(int(N / dec)):
                                if np.mean(np.abs(ind_coefs_DWT_q_enc[ptr:ptr + dec])) != 0:
                                    ind_to_encode.extend(ind_coefs_DWT_q_enc[ptr:ptr + dec])
                                    label_to_encode[i] = 1
                                    coefs_to_encode[i * dec:(i + 1) * dec] = 1
                                ptr += dec
    
                        # Perform entropy coding using Run-Length Encoding (RLE) and Golomb coding
                        code = []
                        cpt_o = 0
                        for k in range(coefs_to_encode_directly):
                            if coefs_to_encode[k] != 0:
                                code.extend(Q.get_code(ind_to_encode[k - cpt_o], allocation_round[k]))
                            else:
                                cpt_o += 1
    
                        binarisation_rle, code_rle, R_opt_binarisation, R1_opt_binarisation, R_opt = RLE_G_enc.rle_golomb_encoding(
                            ind_to_encode[coefs_to_encode_directly - cpt_o:], dico_values, dico_counts, n_value_max
                        )
                        code.extend(code_rle)
    
                        # Perform decoding
                        ind_to_decode = []
                        ptr_code = 0
                        cpt_o = 0
                        for k in range(coefs_to_encode_directly):
                            if coefs_to_encode[k] != 0:
                                ind_to_decode.append(Q.get_inv_code(code[ptr_code:ptr_code + allocation_round[k]], allocation_round[k]))
                                ptr_code += allocation_round[k]
                            else:
                                cpt_o += 1
                        ind_to_decode.extend(RLE_G_dec.rle_golomb_decoding(code[ptr_code:], dico_values, dico_counts, len(ind_to_encode) - coefs_to_encode_directly + cpt_o, n_value_max))
    
                        # Compute real bit rate
                        R_real = len(code)
    
                        # Reconstruct the DWT coefficients
                        coefs_q_dec = np.zeros(len(ind_to_decode), dtype="float")
                        for k in range(len(ind_to_decode)):
                            dynamic = 2 ** (allocation[k] - allocation[0] + 1)
                            #dynamic = 2 ** (allocation_round[k] - allocation_round[0] + 1).astype(float)
                            
                            coefs_q_dec[k] = Q.get_q(ind_to_decode[k], allocation_round[k], dynamic, 0)
    
                        # Restore zero-value bands
                        coefs_DWT_q_dec = np.zeros(N, dtype=float)
                        ptr = 0
                        cpt_dec = 0
                        if dec is not None:
                            for i in range(int(N / dec)):
                                if label_to_encode[i] == 1:
                                    coefs_DWT_q_dec[ptr:ptr + dec] = coefs_q_dec[cpt_dec:cpt_dec + dec]
                                    cpt_dec += dec
                                else:
                                    coefs_DWT_q_dec[ptr:ptr + dec] = [0] * dec
                                ptr += dec
                        else:
                            for i in range(len(label_to_encode)):
                                if label_to_encode[i] == 1:
                                    coefs_DWT_q_dec[ptr:ptr + nb_coeffs_DWT_per_bande[i]] = coefs_q_dec[cpt_dec:cpt_dec + nb_coeffs_DWT_per_bande[i]]
                                    cpt_dec += nb_coeffs_DWT_per_bande[i]
                                else:
                                    coefs_DWT_q_dec[ptr:ptr + nb_coeffs_DWT_per_bande[i]] = [0] * nb_coeffs_DWT_per_bande[i]
                                ptr += nb_coeffs_DWT_per_bande[i]
                        
                        
                        
                        
                        # Scale the reconstructed coefficients
                        coefs_DWT_q_dec *= 2 ** k_DWT
    
                        # Compute SNR and RMSE
                        SNR_test = get_snr(coefs_DWT, coefs_DWT_q_dec)
                        RMSE_test = get_rmse(coefs_DWT, coefs_DWT_q_dec)
    
                        #print(Models[id_model]["family"],Qmin, Qmax, "R={:.2f} b/s".format((R_real + R_h[id_signal][id_phase][w]) / N), end_Na, "RMSE={:.2f}".format(RMSE_test))
                        # Check if the current bit rate does not exceed the total allowed budget
                        if R_real + R_h[id_signal][id_phase][w] <= n_tot:   
                            # Check if the current RMSE is better than the previous model's RMSE
                            if RMSE_test < RMSE_model:
                                
                                # Update the best RMSE found so far for the current model
                                RMSE_model = RMSE_test
                                
                                # If this is the best RMSE found overall, update the best parameters
                                if RMSE_test < RMSE_min:
                                    RMSE_min = RMSE_test
                                    
                                    # Store the best Signal-to-Noise Ratio (SNR) and RMSE
                                    SNR[id_signal][id_phase][w] = SNR_test
                                    RMSE[id_signal][id_phase][w] = RMSE_test
                                    
                                    # Store the actual bit rate used
                                    R_flow[id_signal][id_phase][w] = R_real
                                    R_unused[id_signal][id_phase][w]=n_tot-R_real - R_h[id_signal][id_phase][w]
                                    
                                    # Save the best quantized wavelet coefficients
                                    coefs_DWT_q_dec_best = list(coefs_DWT_q_dec)
                                    
                                    # Reconstruct the signal using inverse wavelet transform
                                    x_rec[id_signal][id_phase][w*N : w*N + N] = W.get_x_rec_DWT(
                                        coefs_DWT_q_dec_best, wave, level
                                    )
                                    
                                    # Store the best model index and corresponding quantization parameters
                                    M[id_signal][id_phase][w] = id_model
                                    Qmax_best = Qmax
                                    Qmin_best = Qmin
                    
                            # If the current RMSE is worse than twice the best model's RMSE, stop testing lower Qmin values
                            if end_Na and RMSE_test > RMSE_model * 2:
                                # Once a suitable Qmax is found, stop testing further Qmin values 
                                # if increasing Qmin significantly worsens the distortion
                                break 
                                                        
                            if end_Na == 0 and Qmin == Qmax_test  / nb_fact:
                                break  # No need to test other Qmin values, as Qmax can still be increased
                        
                        else:  # If the total bit budget is exceeded
                        
                            # If we haven't yet found an optimal Qmax, reduce Qmax and restart testing
                            if end_Na == 0:
                                Qmax_test -= 1
                                end_Na = 1  # A valid Qmin/Qmax pair has been found; no need to increase Qmax further
                            
                           
                    # If an optimal Qmax has been determined, stop testing higher Qmax values
                    if end_Na:
                        break  # Stop increasing Qmax since it would exceed the total bit budget
                    
            # ######################################################################
            # Reconstruct the signal using the best parameters found
            print("id={}, phase={}, w={}, model={}, Qmin={:.2f}, Qmax={:.2f}, R={:.1f}, RMSE={:.1f} V, SNR={:.1f} dB".format(id_signal,id_phase,w,Models[M[id_signal][id_phase][w]]["name"],Qmin_best,Qmax_best,R_flow[id_signal][id_phase][w]/N+R_h[id_signal][id_phase][w]/N,RMSE[id_signal][id_phase][w],SNR[id_signal][id_phase][w]))
        
 
    
#Record the end time
end_time = time.time()

#Calculate the elapsed time
elapsed_time = end_time - start_time

print("time to encode {:.0f} signal(s) ({} phase(s)) of {:.2f} seconde(s): {:.2f} secondes".format(nb_signal,nb_phase,N*nb_w/fs,elapsed_time))    



t = np.linspace(0, (nb_w*N - 1) * (1 / fs), nb_w * N)

  
for id_signal in range(nb_signal):
    
    for id_phase in range(nb_phase):

        #### Reconstructed signal
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot(t, x_real[id_signal][id_phase] / 1000, lw=1, label='x (original)')
        plt.plot(t, x_rec[id_signal][id_phase] / 1000, lw=1, label="x_rec (reconstructed)")
        plt.xlabel('t (s)')
        plt.ylabel('Magnitude x10³')
        plt.legend()
        plt.title(f"Reconstructed Signal: {id_signal}, Phase: {id_phase + 1}, "
                  f"SNR Mean = {np.mean(SNR[id_signal][id_phase]):.2f} dB, "
                  f"RMSE Mean = {np.mean(RMSE[id_signal][id_phase]):.2f} V")
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
    
        #### Reconstruction error
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot(t, (x_real[id_signal][id_phase] - x_rec[id_signal][id_phase]), lw=1, label='x - x_rec (error)')
        plt.xlabel('t (s)')
        plt.ylabel('Magnitude')
        plt.title(f"Reconstruction Error: {id_signal}, Phase: {id_phase + 1}")
        plt.legend()
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()       
        
        largeur_barre=0.6
        plt.figure(figsize=(8,4), dpi=80)
        plt.bar([i for i in range(nb_w)],SNR[id_signal][id_phase], width = largeur_barre,color='b',label="SNR")
        plt.xlabel('index')
        plt.ylabel('SNR (dB)')
        plt.legend()
        plt.title('SNR mean={:.2f} dB'.format(np.mean(SNR[id_signal][id_phase])))
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)  
        
        #### RMSE for each window
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot([t[k] for k in range(0, nb_w * N, N)], RMSE[id_signal][id_phase], '-o', lw=1, label='RMSE')
        plt.xlabel('t (s)')
        plt.ylabel('RMSE')
        plt.title(f"RMSE for Each Window, Mean RMSE = {np.mean(RMSE[id_signal][id_phase]):.0f} V, "
                  f"Signal: {id_signal}, Phase: {id_phase + 1}")
        plt.legend()
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
        
        plt.figure(figsize=(8,4), dpi=80)
        plt.bar([i for i in range(nb_w)],R_h[id_signal][id_phase], width = largeur_barre,color='g',label="R h")
        plt.bar([i for i in range(nb_w)],R_flow[id_signal][id_phase], width = largeur_barre,bottom =R_h[id_signal][id_phase],color='b',label="R")
        plt.bar([i for i in range(nb_w)],R_unused[id_signal][id_phase], width = largeur_barre,bottom =R_flow[id_signal][id_phase]+R_h[id_signal][id_phase],color='c',label="R unused")
        plt.xlabel('index w')
        plt.ylabel('R (bits)')
        plt.legend()
        plt.title('R mean={:.2f} bits'.format(np.mean(R_flow[id_signal][id_phase]+R_h[id_signal][id_phase])))
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)    
        
        #### Model index over time
        yticks_labels = [Models[id_model]["name"] for id_model in Models]
        yticks_positions = np.arange(len(yticks_labels))
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot([t[k] for k in range(0, nb_w * N, N)], M[id_signal][id_phase], 'o', lw=1, label='Model Index')
        plt.xlabel('Window Index')
        plt.ylabel('Model Index')
        plt.legend()
        plt.title(f"Model Selection, Signal: {id_signal}, Phase: {id_phase + 1}")
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.yticks(yticks_positions, yticks_labels)
        plt.show()            
        

  
    
   
