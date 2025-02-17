# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:04:15 2024

@author: presvotscor
"""
import numpy as np
import math as mt

from Context_Arithmetic import Context_Aritmetic_Encoder,Context_Aritmetic_Decoder


class Golomb:
    """
    Class implementing Golomb coding and decoding.
    Golomb coding is a lossless entropy coding technique used for data compression.
    """

    def __init__(self):
        pass  # No initialization parameters needed

    # Method from the "Denoising DCT" paper
    def fix_enc(self, ind, b):
        """
        Fixed-length encoding based on the threshold T = 2^b - 1.

        Parameters:
            ind (int): Input index to encode.
            b (int): Bit length.

        Returns:
            list: Encoded bit sequence.
        """
        T = 2**b - 1
        L = []

        if np.abs(ind) > T:
            Nz = np.floor((np.abs(ind) - 1) / T)  # Compute the number of zero bits
            L.extend([0] * int(Nz))  # Append Nz zeros to the list
            L.append(int(ind - Nz * T))  # Append the remaining value
        else:
            L.append(int(ind))  # If ind is within range, encode directly

        return L  # Return the encoded bit sequence

    def fix_dec(self, code, b):
        """
        Fixed-length decoding.

        Parameters:
            code (list): Encoded bit sequence.
            b (int): Bit length.

        Returns:
            int: Decoded index.
        """
        T = 2**b - 1
        Nz = 0

        # Count the number of leading zeros in the code
        while code[Nz] == 0:
            Nz += 1

        return Nz * T + code[-1]  # Compute the decoded value

    def golomb_cod(self, x, m):
        """
        Perform Golomb encoding.

        Parameters:
            x (int): The input value to encode.
            m (int): Golomb parameter.

        Returns:
            list: Encoded bit sequence.
        """
        c = int(mt.ceil(mt.log(m, 2)))  # Compute the number of bits required
        remin = x % m  # Compute remainder
        quo = int(mt.floor(x / m))  # Compute quotient
        div = int(mt.pow(2, c) - m)  # Compute division adjustment

        bits = []  # Initialize bit list

        # Append '1' for each quotient
        for _ in range(quo):
            bits.append(1)

        # Append '0' after the '1' bits if m is not equal to 1
        if m != 1:
            bits.append(0)

        # Encode the remainder in binary form
        if remin < div:
            b = c - 1
            a = "{0:0" + str(b) + "b}"  # Format binary string with b bits
            bi = a.format(remin)
        else:
            b = c
            a = "{0:0" + str(b) + "b}"
            bi = a.format(remin + div)

        # Convert binary string into list of integers and append
        bits.extend(int(bit) for bit in bi)

        return bits  # Return the encoded bit sequence

    # Golomb parameter m: When m is a power of 2, it is also known as Rice coding
    def creat_dico(self, m, size_dico, pos_E):
        """
        Creates a Golomb-coded dictionary.

        Parameters:
            m (int): Golomb parameter.
            size_dico (int): Maximum size of the dictionary.
            pos_E (int): Special position for the 'E' symbol.

        Returns:
            dict: Dictionary mapping symbols to their encoded values.
        """
        dico = {}
        sym = 0  # Symbol counter
        cpt = 0  # Encoding counter

        while sym <= size_dico + 1:
            if sym == pos_E and cpt == sym:
                dico['E'] = self.golomb_cod(cpt, m)  # Assign Golomb code to the 'E' symbol
                cpt += 1
            else:
                dico[sym] = self.golomb_cod(cpt, m)  # Assign Golomb code to numeric symbols
                sym += 1
                cpt += 1

        return dico  # Return the encoded dictionary



from itertools import accumulate
from Measures import get_frequency,entropy,calculate_entropy,calculate_conditional_entropy#,my_bin,my_inv_bin

class Golomb_encoding(Golomb, Context_Aritmetic_Encoder):
    """
    Class implementing an entropy encoder using Golomb coding with arithmetic coding.
    This encoder supports adaptive probability updates to improve compression efficiency.

    Inherits from:
    - Golomb: Provides Golomb coding methods.
    - Context_Aritmetic_Encoder: Implements arithmetic coding for entropy encoding.

    Attributes:
        adaptive (bool): If True, updates probability distributions dynamically.
    """

    def __init__(self, adaptive):
        """
        Initializes the Golomb encoding class.

        Parameters:
            adaptive (bool): Enables or disables adaptive probability updates.
        """
        Golomb.__init__(self)  # Initialize Golomb coding functionalities

        M = 12  # Precision parameter for arithmetic encoding
        Context_Aritmetic_Encoder.__init__(self, M)  # Initialize arithmetic encoder

        self.adaptive = adaptive  # Store whether adaptation is enabled
        self.reset_Golomb_encoding()  # Reset encoding parameters

    def reset_Golomb_encoding(self):
        """
        Resets the encoding parameters to their initial values.
        This function is used to initialize or reinitialize the entropy encoder.
        """
        self.reset_Context_Aritmetic_Encoder()  # Reset arithmetic encoder state

        # Define probability distributions for encoding binary values (0 or 1)
        self.alphabet_values_p0 = [0, 1]  # Possible symbols when the previous bit was 0
        self.alphabet_values_p1 = [0, 1]  # Possible symbols when the previous bit was 1

        # Initial probability distributions (equal probability for 0 and 1)
        self.occurrence_values_p0 = [1, 1]  # Probability of 0 or 1 given the previous bit was 0
        self.occurrence_values_p1 = [1, 1]  # Probability of 0 or 1 given the previous bit was 1

        # Probability distribution for polarity encoding
        self.alphabet_p = [0, 1]  # Possible polarity values (+ or -)
        self.occurrence_p = [1, 1]  # Equal probability for positive and negative values

        # Compute cumulative probabilities for arithmetic coding
        self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))
        self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))
        self.cumulate_occurrence_p = list(accumulate(self.occurrence_p))

    def encoded_value(self, code_value):
        """
        Entropy encodes a sequence of binary values using adaptive arithmetic coding.

        Parameters:
            code_value (list): The input binary sequence to encode.

        Returns:
            list: Encoded bit sequence.
        """
        code = []
        previous_bit = 1  # Initialize with a default previous bit

        for k in range(len(code_value)):
            if previous_bit == 0:
                if code_value[k] == 0:
                    code_value_k = self.encode_one_symbol(0, self.occurrence_values_p0, self.cumulate_occurrence_values_p0)
                    code.extend(code_value_k)
                    if self.adaptive:
                        self.occurrence_values_p0[0] += 1
                        self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))
                    previous_bit = 0  # Update previous bit
                elif code_value[k] == 1:
                    code_value_k = self.encode_one_symbol(1, self.occurrence_values_p0, self.cumulate_occurrence_values_p0)
                    code.extend(code_value_k)
                    if self.adaptive:
                        self.occurrence_values_p0[1] += 1
                        self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))
                    previous_bit = 1  # Update previous bit
            
            elif previous_bit == 1:
                if code_value[k] == 0:
                    code_value_k = self.encode_one_symbol(0, self.occurrence_values_p1, self.cumulate_occurrence_values_p1)
                    code.extend(code_value_k)
                    if self.adaptive:
                        self.occurrence_values_p1[0] += 1
                        self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))
                    previous_bit = 0  # Update previous bit
                
                elif code_value[k] == 1:
                    code_value_k = self.encode_one_symbol(1, self.occurrence_values_p1, self.cumulate_occurrence_values_p1)
                    code.extend(code_value_k)
                    if self.adaptive:
                        self.occurrence_values_p1[1] += 1
                        self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))
                    previous_bit = 1  # Update previous bit

        return code  # Return the entropy-encoded bit sequence

    def encoded_polarity(self, p):
        """
        Entropy encodes the polarity value (sign of a number).

        Parameters:
            p (int): The polarity value (0 for negative, 1 for positive).

        Returns:
            list: Encoded bit sequence.
        """
        code = []

        if p == 0:
            code_p = self.encode_one_symbol(0, self.occurrence_p, self.cumulate_occurrence_p)
            code.extend(code_p)
            if self.adaptive:
                self.occurrence_p[0] += 1
                self.cumulate_occurrence_p = list(accumulate(self.occurrence_p))

        elif p == 1:
            code_p = self.encode_one_symbol(1, self.occurrence_p, self.cumulate_occurrence_p)
            code.extend(code_p)
            if self.adaptive:
                self.occurrence_p[1] += 1
                self.cumulate_occurrence_p = list(accumulate(self.occurrence_p))

        return code  # Return the entropy-encoded polarity value
    
    def golomb_encoding(self, data, dico, end):
        """
        Performs Golomb encoding on the given data sequence.
    
        Parameters:
            data (list): Input sequence of values to encode.
            dico (dict): Dictionary containing the Golomb codes for each possible value.
            end (bool): If True, removes trailing zeros from the data before encoding.
    
        Returns:
            tuple: (binarisation, code, H, H1, R_opt)
                - binarisation (list): Binary representation of the encoded data.
                - code (list): Final compressed bit sequence.
                - H (float): Shannon entropy of the binarized values.
                - H1 (float): Conditional entropy of the binarized values.
                - R_opt (float): Optimal bit rate based on entropy calculations.
        """
    
        self.reset_Golomb_encoding()  # Reset encoding parameters to initial state
    
        binarisation_value = []  # Stores the binary representation of encoded values
        binarisation_p = []  # Stores the binary representation of polarity bits
        N = len(data)
        binarisation = []  # Complete binary sequence before entropy encoding
        code = []  # Final encoded bitstream
    
        i = 1  # Counter to track trailing zeros
        if end:
            # If 'end' is True, remove trailing zeros from the data
            while i <= len(data) and data[-i] == 0:
                i += 1
            # Remaining data to be encoded after removing trailing zeros
            # print("data[:N-i]", data[:N-i+1], len(data[:N-i+1]))
    
        # Encode each value in the data (excluding trailing zeros if 'end' is True)
        for value in data[:N - i + 1]:
            # Get the Golomb code from the dictionary
            code_value = dico[np.abs(value)]
            binarisation.extend(code_value)  # Append binary representation
    
            binarisation_value.extend(code_value)  # Store values separately
            # print("value", value, code_value, len(code_value))
    
            # Apply entropy coding to code_value using arithmetic coding
            code_ar = self.encoded_value(code_value)
            # print("code_value", len(code_value), ", code_ar", len(code_ar))
            code.extend(code_ar)  # Append the entropy-encoded sequence
    
            # Encode the polarity bit if the value is non-zero
            if value != 0:
                code_p = int((np.sign(value) + 1) / 2)  # Convert sign to binary (0 for negative, 1 for positive)
                binarisation.append(code_p)
                binarisation_p.append(code_p)
    
                # Apply entropy coding to the polarity bit
                code.extend(self.encoded_polarity(code_p))
    
        # If trailing zeros were removed, add the "E" (end) symbol encoding
        if i != 1:
            code_E = dico["E"]  # Get end symbol encoding from dictionary
            binarisation.extend(code_E)
            code.extend(self.encoded_value(code_E))
    
            binarisation_value.extend(code_E)  # Store encoded end marker separately
    
        # Finalize the arithmetic encoding by appending termination bits
        code_end = self.finish(self.l, self.follow)
        code.extend(code_end)
    
        # print("len_bin={}, len_code={}".format(len(binarisation), len(code)))
    
        # Compute entropy metrics for the encoded data
        _, proba_data = get_frequency(np.abs(data))  # Compute symbol probability distribution
        R_opt = entropy(proba_data) * len(data) + len(data) - data.count(0.)  # Optimal entropy-based bit rate
    
        H = calculate_entropy(binarisation_value) * len(binarisation_value) + calculate_entropy(binarisation_p) * len(binarisation_p)
        H1 = calculate_conditional_entropy(binarisation_value) * len(binarisation_value) + calculate_entropy(binarisation_p) * len(binarisation_p)
    
        # print("Golomb R={:.0f} b, H={:.1f} b, H1={:.1f} b".format(len(code), H, H1))
    
        return binarisation, code, H, H1, R_opt


class Golomb_decoding(Golomb, Context_Aritmetic_Decoder):
    """
    Golomb Decoding with Adaptive Arithmetic Coding

    This class implements Golomb decoding combined with context-based arithmetic decoding. 
    It supports adaptive probability updates for symbol occurrences, allowing efficient entropy decoding.

    Attributes:
        adaptive (bool): Indicates whether adaptive probability updates are used.
    """

    def __init__(self, adaptive):
        """
        Initializes the Golomb decoder and its arithmetic decoder component.

        Args:
            adaptive (bool): Enables adaptive probability updates if True.
        """
        Golomb.__init__(self)  # Initialize Golomb base class
        
        M = 12  # Precision for arithmetic decoding
        Context_Aritmetic_Decoder.__init__(self, M)  # Initialize arithmetic decoder

        self.adaptive = adaptive  # Enable or disable adaptive updates

    def reset_Golomb_decoding(self):
        """
        Resets the state of the Golomb decoder, including context-based probabilities.
        """
        self.reset_Context_Aritmetic_Decoder()  # Reset arithmetic decoder state

        # Define binary alphabet for entropy decoding
        self.alphabet_values_p0 = [0, 1]  # Binary values for sequences starting with 0
        self.alphabet_values_p1 = [0, 1]  # Binary values for sequences starting with 1

        # Initialize occurrence counts (equal probability assumption at start)
        self.occurrence_values_p0 = [1, 1]  # Probability distribution for values given previous bit is 0
        self.occurrence_values_p1 = [1, 1]  # Probability distribution for values given previous bit is 1

        self.alphabet_p = [0, 1]  # 0: negative polarity, 1: positive polarity
        self.occurrence_p = [1, 1]  # Probability of polarity occurrences [+,-]

        # Compute cumulative distributions for arithmetic decoding
        self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))
        self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))
        self.cumulate_occurrence_p = list(accumulate(self.occurrence_p))

    def decoded_value(self, code, previous_bit):
        """
        Decodes a single binary value using adaptive arithmetic decoding.

        Args:
            code (list): Encoded bit sequence.
            previous_bit (int): Previous bit (used for context-based decoding).

        Returns:
            int: Decoded binary value (0 or 1).
        """
        if previous_bit == 0:
            # Decode using context where previous bit was 0
            bin_value = self.decode_one_symbol(code, self.alphabet_values_p0, self.occurrence_values_p0, self.cumulate_occurrence_values_p0)
            if self.adaptive:
                self.occurrence_values_p0[bin_value] += 1
                self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))

        elif previous_bit == 1:
            # Decode using context where previous bit was 1
            bin_value = self.decode_one_symbol(code, self.alphabet_values_p1, self.occurrence_values_p1, self.cumulate_occurrence_values_p1)
            if self.adaptive:
                self.occurrence_values_p1[bin_value] += 1
                self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))

        return bin_value

    def decoded_polarity(self, code):
        """
        Decodes the polarity bit of a Golomb-coded value.

        Args:
            code (list): Encoded bit sequence.

        Returns:
            int: Decoded polarity bit (0 for negative, 1 for positive).
        """
        bin_p = self.decode_one_symbol(code, self.alphabet_p, self.occurrence_p, self.cumulate_occurrence_p)
        if self.adaptive:
            self.occurrence_p[bin_p] += 1
            self.cumulate_occurrence_p = list(accumulate(self.occurrence_p))

        return bin_p

    def golomb_decoding(self, code, dico, N):
        """
        Decodes a sequence of symbols from a Golomb-coded bitstream.

        Args:
            code (list): Encoded bit sequence.
            dico (dict): Golomb dictionary mapping values to codes.
            N (int): Expected length of the decoded sequence.

        Returns:
            list: Decoded sequence of integer values.
        """
        self.code = code
        self.reset_Golomb_decoding()  # Reset decoder state
        self.ini_codeword(self.code)  # Initialize decoding process
        sym = []  # Stores the decoded symbols
        code_value = []  # Stores temporary encoded sequences before matching

        previous_bit = 1  # Default starting bit for decoding

        while len(sym) < N:
            # Decode next binary value using context-based arithmetic decoding
            bin_value = self.decoded_value(self.code, previous_bit)
            previous_bit = bin_value  # Update context for next decoding step

            code_value.append(bin_value)  # Append decoded binary sequence

            # Check if the current binary sequence matches any entry in the dictionary
            for element in dico:
                if dico[element] == code_value:
                    if element == "E":  
                        # "E" symbol marks the end of transmission -> Fill remaining space with zeros
                        sym.extend([0] * (N - len(sym)))
                        code_value = []
                    elif element == 0:
                        # If the symbol is zero, append it to the output
                        sym.append(element)
                        code_value = []
                    else:
                        # If it's a nonzero value, decode the polarity
                        bin_p = self.decoded_polarity(self.code)
                        sign = bin_p * 2 - 1  # Convert binary polarity back to +1/-1
                        sym.append(sign * element)
                        code_value = []

                    previous_bit = 1  # Reset previous bit after processing a symbol
                    break  # Exit loop since we found a valid symbol

        return sym  # Return the fully decoded sequence

    
    
    
    
    
    
    
class RLE_Golomb_encoding(Golomb, Context_Aritmetic_Encoder):
    """
    Run-Length Encoding (RLE) with Golomb and Context-Based Arithmetic Encoding

    This class extends Golomb encoding by integrating Run-Length Encoding (RLE) 
    and context-based arithmetic encoding. It supports adaptive probability updates 
    to improve compression efficiency.

    Attributes:
        adaptive (bool): Enables adaptive probability updates if True.
    """

    def __init__(self, adaptive):
        """
        Initializes the RLE Golomb encoder and its arithmetic encoding components.

        Args:
            adaptive (bool): Enables adaptive probability updates if True.
        """
        Golomb.__init__(self)  # Initialize Golomb base class

        M = 12  # Precision for arithmetic encoding
        Context_Aritmetic_Encoder.__init__(self, M)  # Initialize arithmetic encoder

        self.adaptive = adaptive  # Enable or disable adaptive updates

    def reset_RLE_Golomb_encoding(self):
        """
        Resets the state of the RLE Golomb encoder, including probability distributions.
        """
        self.reset_Context_Aritmetic_Encoder()  # Reset arithmetic encoder state

        # Define binary alphabets for entropy encoding
        self.alphabet_values_p0 = [0, 1]  # Binary values given previous bit is 0
        self.alphabet_values_p1 = [0, 1]  # Binary values given previous bit is 1

        # Initialize probability distributions for values
        self.occurrence_values_p0 = [1, 1]  # Probability distribution for values given previous bit is 0
        self.occurrence_values_p1 = [1, 1]  # Probability distribution for values given previous bit is 1

        # Define binary alphabets for run-length encoding (RLE)
        self.alphabet_count_p0 = [0, 1]  # RLE count probabilities given previous bit is 0
        self.alphabet_count_p1 = [0, 1]  # RLE count probabilities given previous bit is 1

        self.occurrence_count_p0 = [1, 1]  # Probability distribution for RLE counts given previous bit is 0
        self.occurrence_count_p1 = [1, 1]  # Probability distribution for RLE counts given previous bit is 1

        # Define binary alphabet for polarity encoding
        self.alphabet_p = [0, 1]  # 0: negative polarity, 1: positive polarity
        self.occurrence_p = [1, 1]  # Probability of polarity occurrences [+,-]

        # Compute cumulative distributions for arithmetic encoding
        self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))
        self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))

        self.cumulate_occurrence_count_p0 = list(accumulate(self.occurrence_count_p0))
        self.cumulate_occurrence_count_p1 = list(accumulate(self.occurrence_count_p1))

        self.cumulate_occurrence_p = list(accumulate(self.occurrence_p))

    def encoded_value(self, code_value):
        """
        Encodes a sequence of binary values using adaptive arithmetic encoding.

        Args:
            code_value (list): List of binary values to encode.

        Returns:
            list: Encoded bit sequence.
        """
        code = []
        previous_bit = 1  # Start with an initial bit context

        for k in range(len(code_value)):
            if previous_bit == 0:
                if code_value[k] == 0:
                    code_value_k = self.encode_one_symbol(0, self.occurrence_values_p0, self.cumulate_occurrence_values_p0)
                    code.extend(code_value_k)
                    if self.adaptive:
                        self.occurrence_values_p0[0] += 1
                        self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))
                    previous_bit = 0
                elif code_value[k] == 1:
                    code_value_k = self.encode_one_symbol(1, self.occurrence_values_p0, self.cumulate_occurrence_values_p0)
                    code.extend(code_value_k)
                    if self.adaptive:
                        self.occurrence_values_p0[1] += 1
                        self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))
                    previous_bit = 1
            elif previous_bit == 1:
                if code_value[k] == 0:
                    code_value_k = self.encode_one_symbol(0, self.occurrence_values_p1, self.cumulate_occurrence_values_p1)
                    code.extend(code_value_k)
                    if self.adaptive:
                        self.occurrence_values_p1[0] += 1
                        self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))
                    previous_bit = 0
                elif code_value[k] == 1:
                    code_value_k = self.encode_one_symbol(1, self.occurrence_values_p1, self.cumulate_occurrence_values_p1)
                    code.extend(code_value_k)
                    if self.adaptive:
                        self.occurrence_values_p1[1] += 1
                        self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))
                    previous_bit = 1

        return code

    def encoded_count(self, code_count):
        """
        Encodes a sequence of run-length counts using adaptive arithmetic encoding.

        Args:
            code_count (list): List of binary count values to encode.

        Returns:
            list: Encoded bit sequence.
        """
        code = []
        previous_bit = 1  # Start with an initial bit context

        for k in range(len(code_count)):
            if previous_bit == 0:
                if code_count[k] == 0:
                    code_count_k = self.encode_one_symbol(0, self.occurrence_count_p0, self.cumulate_occurrence_count_p0)
                    code.extend(code_count_k)
                    if self.adaptive:
                        self.occurrence_count_p0[0] += 1
                        self.cumulate_occurrence_count_p0 = list(accumulate(self.occurrence_count_p0))
                    previous_bit = 0
                elif code_count[k] == 1:
                    code_count_k = self.encode_one_symbol(1, self.occurrence_count_p0, self.cumulate_occurrence_count_p0)
                    code.extend(code_count_k)
                    if self.adaptive:
                        self.occurrence_count_p0[1] += 1
                        self.cumulate_occurrence_count_p0 = list(accumulate(self.occurrence_count_p0))
                    previous_bit = 1
            elif previous_bit == 1:
                if code_count[k] == 0:
                    code_count_k = self.encode_one_symbol(0, self.occurrence_count_p1, self.cumulate_occurrence_count_p1)
                    code.extend(code_count_k)
                    if self.adaptive:
                        self.occurrence_count_p1[0] += 1
                        self.cumulate_occurrence_count_p1 = list(accumulate(self.occurrence_count_p1))
                    previous_bit = 0
                elif code_count[k] == 1:
                    code_count_k = self.encode_one_symbol(1, self.occurrence_count_p1, self.cumulate_occurrence_count_p1)
                    code.extend(code_count_k)
                    if self.adaptive:
                        self.occurrence_count_p1[1] += 1
                        self.cumulate_occurrence_count_p1 = list(accumulate(self.occurrence_count_p1))
                    previous_bit = 1

        return code

    def encoded_polarity(self, p):
        """
        Encodes the polarity bit of a Golomb-coded value.

        Args:
            p (int): Polarity bit (0 for negative, 1 for positive).

        Returns:
            list: Encoded bit sequence.
        """
        code = []
        code_p = self.encode_one_symbol(p, self.occurrence_p, self.cumulate_occurrence_p)
        code.extend(code_p)

        if self.adaptive:
            self.occurrence_p[p] += 1
            self.cumulate_occurrence_p = list(accumulate(self.occurrence_p))

        return code
    
    def run_length_encoding(self, data, n_value_max):
        """
        Performs run-length encoding (RLE) on the input data.
    
        Args:
            data (list): The input sequence to be encoded.
            n_value_max (int): The maximum allowed quantized value.
    
        Returns:
            list: A list of tuples (encoded value, polarity, zero count).
                  Each tuple contains:
                  - The Golomb-coded absolute value
                  - The sign of the value (0 for negative, 1 for positive)
                  - The number of consecutive zeros before this value
        """
        rle = []  # List to store run-length encoded values
        count = 0  # Counter for consecutive zeros
    
        for k in range(len(data)):
            value = data[k]
            
            if value == 0 and k < len(data) - 1:
                count += 1  # Increment zero count if the value is zero
            else:
                # Encode the absolute value using fixed-length Golomb encoding
                v = self.fix_enc(np.abs(value), n_value_max)
                # Store the tuple (encoded value, polarity, zero count)
                rle.append((v, int((np.sign(value) + 1) / 2), count))
                count = 0  # Reset zero count
    
        return rle
    
    
    def rle_golomb_encoding(self, data, dico_value, dico_count, n_value_max):
        """
        Performs Run-Length Encoding (RLE) combined with Golomb coding and entropy encoding.
    
        Args:
            data (list): The input sequence to be encoded.
            dico_value (dict): Dictionary mapping quantized values to their binary Golomb codes.
            dico_count (dict): Dictionary mapping run lengths (zero counts) to their binary Golomb codes.
            n_value_max (int): The maximum allowed quantized value.
    
        Returns:
            tuple: A tuple containing:
                - binarisation (list): The raw bit sequence before entropy encoding.
                - code (list): The final encoded bit sequence.
                - H (float): Entropy of the encoded values.
                - H1 (float): Conditional entropy of the encoded values.
                - R_opt (float): Theoretical optimal entropy rate.
        """
        self.reset_RLE_Golomb_encoding()  # Reset state for encoding
    
        rle = self.run_length_encoding(data, n_value_max)  # Apply RLE to input data
    
        # Lists for entropy computation
        binarisation_value = []
        binarisation_count = []
        binarisation = []  # Full binarized sequence before entropy encoding
        code = []  # Final encoded sequence
        nb_s = 0  # Counter for the number of significant symbols
    
        for value, sign, count in rle:
            if len(value) == 1 and value[0] == 0:
                # If the value is zero, use the special Golomb code for "E" (end marker)
                code_E = dico_value["E"]
                binarisation.extend(code_E)
                code.extend(self.encoded_value(code_E))
                binarisation_value.extend(code_E)
            else:
                # Encode the quantized values
                for k in range(len(value)):
                    code_value = dico_value[value[k]]
                    binarisation.extend(code_value)
                    code.extend(self.encoded_value(code_value))
                    binarisation_value.extend(code_value)
    
                # Encode the sign (0 for negative, 1 for positive)
                binarisation.append(sign)
                code.extend(self.encoded_polarity(sign))
    
                # Encode the zero run-length count
                code_count = dico_count[count]
                binarisation.extend(code_count)
                code.extend(self.encoded_count(code_count))
                binarisation_count.extend(code_count)
    
                nb_s += 1  # Increment the number of significant symbols
    
        # Finalize the encoded sequence with an end marker
        code.extend(self.finish(self.l, self.follow))
    
        # Compute entropy measures for performance evaluation
        _, proba_data = get_frequency(np.abs(data))
        R_opt = entropy(proba_data) * len(data) + len(data) - data.count(0.)
    
        H = (
            calculate_entropy(binarisation_value) * len(binarisation_value)
            + calculate_entropy(binarisation_count) * len(binarisation_count)
            + nb_s
        )
    
        H1 = (
            calculate_conditional_entropy(binarisation_value) * len(binarisation_value)
            + calculate_conditional_entropy(binarisation_count) * len(binarisation_count)
            + nb_s
        )
    
        return binarisation, code, H, H1, R_opt
    
    



class RLE_Golomb_decoding(Golomb, Context_Aritmetic_Decoder):
    """
    Class implementing Run-Length Encoding (RLE) combined with Golomb decoding and arithmetic decoding.

    This decoder restores quantized values, their polarities, and run-lengths
    from an encoded bitstream that was compressed using RLE and Golomb coding.
    """

    def __init__(self, adaptive):
        """
        Initializes the RLE Golomb decoder.

        Args:
            adaptive (bool): If True, the decoder updates symbol probabilities dynamically.
        """
        Golomb.__init__(self)  # Initialize the Golomb base class
        M = 12  # Precision for arithmetic coding
        Context_Aritmetic_Decoder.__init__(self, M)  # Initialize arithmetic decoder with precision M
        self.adaptive = adaptive  # Enable or disable adaptive probability updates

    def reset_RLE_Golomb_decoding(self):
        """
        Resets the decoding context, including probability distributions for adaptive arithmetic decoding.
        """
        self.reset_Context_Aritmetic_Decoder()

        # Initialize probability tables for value decoding
        self.alphabet_values_p0 = [0, 1]
        self.alphabet_values_p1 = [0, 1]
        self.occurrence_values_p0 = [1, 1]  # Initial probability for values given previous bit = 0
        self.occurrence_values_p1 = [1, 1]  # Initial probability for values given previous bit = 1

        # Initialize probability tables for run-length decoding
        self.alphabet_count_p0 = [0, 1]
        self.alphabet_count_p1 = [0, 1]
        self.occurrence_count_p0 = [1, 1]  # Initial probability for run-lengths given previous bit = 0
        self.occurrence_count_p1 = [1, 1]  # Initial probability for run-lengths given previous bit = 1

        # Initialize polarity probabilities (0: negative, 1: positive)
        self.alphabet_p = [0, 1]
        self.occurrence_p = [1, 1]

        # Compute cumulative distributions for adaptive arithmetic decoding
        self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))
        self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))
        self.cumulate_occurrence_count_p0 = list(accumulate(self.occurrence_count_p0))
        self.cumulate_occurrence_count_p1 = list(accumulate(self.occurrence_count_p1))
        self.cumulate_occurrence_p = list(accumulate(self.occurrence_p))

    def decoded_value(self, code, previous_bit):
        """
        Decodes a binary value using arithmetic decoding.

        Args:
            code (list): The bitstream being decoded.
            previous_bit (int): The previous decoded bit, used for context-based probability.

        Returns:
            int: The decoded binary value (0 or 1).
        """
        if previous_bit == 0:
            bin_value = self.decode_one_symbol(code, self.alphabet_values_p0, self.occurrence_values_p0, self.cumulate_occurrence_values_p0)
            if self.adaptive:
                self.occurrence_values_p0[bin_value] += 1
                self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))
        else:
            bin_value = self.decode_one_symbol(code, self.alphabet_values_p1, self.occurrence_values_p1, self.cumulate_occurrence_values_p1)
            if self.adaptive:
                self.occurrence_values_p1[bin_value] += 1
                self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))
        return bin_value

    def decoded_count(self, code, previous_bit):
        """
        Decodes a run-length count using arithmetic decoding.

        Args:
            code (list): The bitstream being decoded.
            previous_bit (int): The previous decoded bit, used for context-based probability.

        Returns:
            int: The decoded run-length count.
        """
        if previous_bit == 0:
            bin_count = self.decode_one_symbol(code, self.alphabet_count_p0, self.occurrence_count_p0, self.cumulate_occurrence_count_p0)
            if self.adaptive:
                self.occurrence_count_p0[bin_count] += 1
                self.cumulate_occurrence_count_p0 = list(accumulate(self.occurrence_count_p0))
        else:
            bin_count = self.decode_one_symbol(code, self.alphabet_count_p1, self.occurrence_count_p1, self.cumulate_occurrence_count_p1)
            if self.adaptive:
                self.occurrence_count_p1[bin_count] += 1
                self.cumulate_occurrence_count_p1 = list(accumulate(self.occurrence_count_p1))
        return bin_count

    def decoded_polarity(self, code):
        """
        Decodes a polarity bit (sign) using arithmetic decoding.

        Args:
            code (list): The bitstream being decoded.

        Returns:
            int: 0 for negative, 1 for positive.
        """
        bin_p = self.decode_one_symbol(code, self.alphabet_p, self.occurrence_p, self.cumulate_occurrence_p)
        if self.adaptive:
            self.occurrence_p[bin_p] += 1
            self.cumulate_occurrence_p = list(accumulate(self.occurrence_p))
        return bin_p

    def rle_golomb_decoding(self, code, dico_value, dico_count, N, n_value_max):
        """
        Decodes a bitstream compressed using Run-Length Encoding (RLE) and Golomb coding.

        Args:
            code (list): The encoded bitstream.
            dico_value (dict): Dictionary mapping encoded values to their original values.
            dico_count (dict): Dictionary mapping encoded run-lengths to their original counts.
            N (int): Expected number of decoded values.
            n_value_max (int): Maximum possible quantized value.

        Returns:
            list: The fully decoded sequence.
        """
        self.code = code
        self.reset_RLE_Golomb_decoding()
        self.ini_codeword(self.code)

        sym = []  # List to store the decoded values
        value = True  # Flag to switch between value decoding and run-length decoding
        code_value_fix = []  # Buffer for fixed-length values
        code_value = []  # Buffer for variable-length value decoding
        code_count = []  # Buffer for run-length decoding

        previous_bit_value = 1
        previous_bit_count = 1

        while len(sym) < N:
            if value:
                # Decode value
                bin_value = self.decoded_value(self.code, previous_bit_value)
                previous_bit_value = bin_value
                code_value.append(bin_value)

                for v in dico_value:
                    if dico_value[v] == code_value:
                        if v == "E":  # End-of-sequence marker
                            sym.extend([0] * (N - len(sym)))
                            code_value = []
                            value = False
                        elif v == 0:
                            code_value_fix.append(0)
                            code_value = []
                            previous_bit_value = 1
                            value = True
                        else:
                            # Decode polarity (sign)
                            bin_p = self.decoded_polarity(self.code)
                            sign = bin_p * 2 - 1
                            code_value_fix.append(v)
                            ind = self.fix_dec(code_value_fix, n_value_max)
                            code_value_fix = []
                            code_value = []
                            previous_bit_value = 1
                            value = False
                        break
            else:
                # Decode run-length count
                bin_count = self.decoded_count(self.code, previous_bit_count)
                previous_bit_count = bin_count
                code_count.append(bin_count)

                for c in dico_count:
                    if dico_count[c] == code_count:
                        sym.extend([0] * c + [ind * sign])
                        code_count = []
                        previous_bit_count = 1
                        value = True
                        break

        return sym

# Test script for Golomb encoding
if __name__ == "__main__":

    # Import required modules
    from get_test_signal import get_RTE_signal as get_signal  # Function to generate test signals
    from DWT import Wavelet  # Import wavelet processing class
    from Quantization import Quantizer  # Import quantization class
    from Normalize import normalize  # Import normalization function
    
    # Define parameters
    fs = 6400  # Sampling frequency in Hz
    fn = 50    # Nominal frequency of the signal in Hz
    N = 128    # Number of samples per window
    T = 0.02   # Total signal duration in seconds

    n = 8  # Number of bits per coefficient for quantization

    # Select test signal parameters
    id_signal = 1  # ID of the signal to encode
    id_phase = 0   # Signal phase (u1, u2, u3, i1, i2, i3)
    id_w = 20      # Window index within the signal

    # Retrieve the signal for the given parameters
    x_test = get_signal(id_signal)[id_phase][id_w * N:(id_w + 1) * N]

    # Choose the wavelet function for decomposition
    wave = "coif5"  
    level = int(np.log2(N))  # Compute the wavelet decomposition level based on signal length

    # Initialize classes for wavelet processing and quantization
    W = Wavelet()
    Q = Quantizer()
    
    # Perform wavelet decomposition to extract coefficients
    coefs_DWT = W.get_coefs_DWT(x_test, wave, level)

    # Normalize the wavelet coefficients
    coefs_DWT_n, kx = normalize(coefs_DWT)

    # Initialize an array for quantized coefficients
    x = np.zeros(N)

    # Quantize the wavelet coefficients using a uniform quantizer
    for k in range(N):
        dynamic = 2  # Set a fixed dynamic range for quantization
        x[k] = Q.get_ind(coefs_DWT_n[k], n, dynamic, 0)

    # Convert the quantized coefficients into a list format
    x = list(x)

    # Compute the entropy of the quantized coefficients
    alphabet_x, probability_x = get_frequency(x)
    Hx = entropy(probability_x)  # Shannon entropy measure

    ##### Golomb Encoding Test #####
    G = Golomb()  # Initialize Golomb coding class

    ############################### Run-Length Encoding + Golomb Coding ###############################
    adaptive = True  # Enable adaptive encoding
    RLE_G_enc = RLE_Golomb_encoding(adaptive)  # Initialize RLE + Golomb encoder
    RLE_G_dec = RLE_Golomb_decoding(adaptive)  # Initialize RLE + Golomb decoder

    # Define Golomb coding parameters
    m_values = 3  # Golomb parameter for value encoding
    m_counts = 2  # Golomb parameter for count encoding
    pos_E = 3  # End-of-sequence marker position
    n_value_max = 7  # Maximum number of bits per quantized value
    size_dico = 2 ** (n_value_max)  # Dictionary size for values

    # Create Golomb coding dictionaries for run-length encoding
    dico_counts_rle = G.creat_dico(m_counts, N, None)  # Dictionary for counts
    dico_values_rle = G.creat_dico(m_values, size_dico, pos_E)  # Dictionary for values

    # Apply RLE + Golomb encoding
    binarisation_x_rle, code_x_rle, R_opt_binarisation_rle, R1_opt_binarisation_rle, R_opt_data = \
        RLE_G_enc.rle_golomb_encoding(x, dico_values_rle, dico_counts_rle, n_value_max)

    # Decode the RLE + Golomb encoded data
    x_rec_rle = RLE_G_dec.rle_golomb_decoding(code_x_rle, dico_values_rle, dico_counts_rle, len(x), n_value_max)

    # Verify lossless decoding
    if np.mean(x) == np.mean(x_rec_rle):
        print("RLE decoding successful")
    else:
        print("RLE decoding failed")
        print("Original: ", x)
        print("Decoded:  ", list(x_rec_rle))

    ################################# Standard Golomb Encoding #################################
    G_enc = Golomb_encoding(adaptive)  # Initialize standard Golomb encoder
    G_dec = Golomb_decoding(adaptive)  # Initialize standard Golomb decoder
    end = False  # No explicit end marker

    # Create dictionary for standard Golomb encoding
    dico_values = G.creat_dico(3, 2**11, 10)

    # Apply standard Golomb encoding
    binarisation_x, code_x, R_opt_binarisation, R1_opt_binarisation, R_opt_data = \
        G_enc.golomb_encoding(x, dico_values, end)

    # Decode the Golomb encoded data
    x_dec = G_dec.golomb_decoding(code_x, dico_values, len(x))

    # Verify lossless decoding
    if np.mean(x) == np.mean(x_dec):
        print("Golomb decoding successful")
    else:
        print("Golomb decoding failed")
        print("Original: ", x[:30], np.mean(abs(x)))
        print("Decoded:  ", x_dec[:30], np.mean(np.abs(x_dec)))

    ### Display Encoding Efficiency ###
    print("H data                    = {:.2f} bits - Entropy of the quantized signal".format(R_opt_data))
    
    print("R binarization (RLE)  / opt = {:.0f} / {:.2f} bits - Bit count after binarization using RLE, compared to the entropy of the binary source".format(len(binarisation_x_rle), R_opt_binarisation_rle))
    
    print("R1 binarization (RLE) / opt = {:.0f} / {:.2f} bits - Bit count after binarization using RLE with a Markov source, compared to the entropy of the binary source with a first-order Markov model".format(len(code_x_rle), R1_opt_binarisation_rle))
    
    print("R binarization        / opt = {:.0f} / {:.2f} bits - Bit count after binarization without RLE, compared to the entropy of the binary source".format(len(binarisation_x), R_opt_binarisation))
    
    print("R1 binarization       / opt = {:.0f} / {:.2f} bits - Bit count after binarization without RLE using a Markov source, compared to the entropy of the binary source with a first-order Markov model".format(len(code_x), R1_opt_binarisation))
