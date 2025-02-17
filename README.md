# Overcomplete Hybrid Dictionaries

This repository provides a Python-based implementation of the method described in the article:  
**Sabarimalai Manikandan, M., Kamwa, I., and Samantaray, S. R. (2015).**  
*"Simultaneous denoising and compression of power system disturbances using sparse representation on overcomplete hybrid dictionaries."*  
**IET Generation, Transmission & Distribution, 9(11), 1077–1088.**  
[doi:10.1049/iet-gtd.2014.0806](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-gtd.2014.0806)

The code is provided as an open-source implementation since the original article did not include one. This repository can serve as a starting point for researchers and engineers interested in experimenting with or extending the method’s capabilities.

---

## Main Scripts

This repository includes two main scripts:  
- **OHD_Rate_Constraint:** Implements the method with a fixed bit rate constraint.  
- **OHD_Quality_Constraint:** Implements the method with a fixed quality constraint.  

Either script can be run directly after downloading the full OHD code.

---

## Key Parameters

### Data Source and Parameters 
By default, the code uses 12 three-phase voltage signals from the [Data_S](https://github.com/rte-france/digital-fault-recording-database) dataset. These signals are selected because they correspond to known faults. Each signal is one second long and sampled at 6400 Hz, resulting in 100 non-overlapping 20 ms windows per signal.

- **Number of signals:** The number of signals to encode is controlled by `nb_signal`.  
- **Number of phases:** Specified by `nb_phase`:  
  - `nb_phase=3` processes only the three voltage phases.  
  - `nb_phase=6` includes both the three voltage phases and three current phases for all 12 signals.  
- **Number of windows (`nb_w`):** By default, the first 50 windows of each signal are encoded. This can be adjusted to process more or fewer windows per signal.
- **Window Size (`N`):** Each window is set to 128 samples by default. This can be adjusted by modifying `N` in the script.  

### Key Parameter for Rate Constraint

- **Maximum Bit Rate (`n_tot`):** The total bit rate allocated for encoding each window is 128 bits (equivalent to 1 bit per sample) by default. You can modify `n_tot` to explore different encoding rates.

### Key Parameters for Quality Constraint

- **Quality Constraint (`quality`):** The default quality threshold is an RMSE of 200 V. Adjusting `quality` allows you to experiment with different levels of reconstruction accuracy and compression performance.  
- **Available Metrics:** The code supports three quality metrics derived from the L2 norm: RMSE, MSE, and -SNR.

---

## Description of Reimplemented Method for Quality Constraint

The approach simultaneously denoises and compresses power system disturbances by leveraging sparse representation over a hybrid dictionary that combines impulse, discrete cosine, and discrete sine bases. By using overcomplete dictionaries, the method reduces block boundary artifacts and facilitates direct estimation of power quantities from the coefficients associated with sinusoidal components.

### Dictionary Construction
The dictionary, $\boldsymbol{D}$, is a concatenation of three matrices:
```math
\boldsymbol{D} = \left[\begin{array}{lll}
\boldsymbol{C} & \mid & \boldsymbol{S} \mid \boldsymbol{I}
\end{array}\right]_{N \times 3N}
```

- **Cosine Matrix ($\boldsymbol{C}$)**: A set of sampled discrete cosine waveforms ($N \times N$), where
```math
\left[\boldsymbol{C}\right]_{ij} = \sqrt{\frac{2}{N}} \cdot \varepsilon_{i} \cdot \cos\left(\frac{\pi(2j+1)i}{2N}\right),\quad i=0,\dots,N-1, \ j=0,\dots,N-1
```

  $\varepsilon_i = \frac{1}{\sqrt{2}}$ for $i = 0$, otherwise $\varepsilon_i = 1$.
- **Sine Matrix ($\boldsymbol{S}$)**: A set of sampled discrete sine waveforms ($N \times N$), where
```math
\left[\boldsymbol{S}\right]_{ij} = \sqrt{\frac{2}{N}} \cdot \varepsilon_{i} \cdot \sin\left(\frac{\pi(2j+1)(i+1)}{2N}\right),\quad i=0,\dots,N-1.
```
$\varepsilon_i = \frac{1}{\sqrt{2}}$ for $i = 0$, otherwise $\varepsilon_i = 1$.
- **Impulse Matrix ($\boldsymbol{I}$)**: An identity matrix ($N \times N$) representing discrete impulses.
### Sparse Approximation and Matching Pursuit
The sparse representation is achieved using a matching pursuit algorithm that iteratively selects the vectors of $\boldsymbol{D}=(\boldsymbol{d} \_1,\dots,\boldsymbol{d} \_{3N})^T$ that best reduce the approximation error. This can be expressed as
```math
MSE = \frac{1}{N} \left\| \mathbf{x} - \sum_{j=1}^{K} \widehat{\alpha}_j \mathbf{d}_{\widehat{i}_j} \right\|^2,
```
where 

```math
\widehat{\alpha}_1, \ldots, \widehat{\alpha}_K, \widehat{i}_1, \ldots, \widehat{i}_K=\arg \min_{\alpha_1, \ldots, \alpha_K, i_1, \ldots, i_K} \frac{1}{N} \left\| \mathbf{x} - \sum_{j=1}^{K} \alpha_j \mathbf{d}_{i_j} \right\|^2.
```
The process continues until $MSE \leqslant MSE_{\text{max}}$.

### Coefficient Reordering
Since the dictionary is not orthogonal, the coefficients $\widehat{\alpha}_1, \ldots, \widehat{\alpha}_K$ are sorted in descending order
```math
\pi : \{1, \ldots, K\} \to \{\widehat{\alpha}_1, \ldots, \widehat{\alpha}_K\}
```

so that $\pi(1) < \pi(2) < \ldots < \pi(K)$. The corresponding reordered coefficients are
```math
\{\alpha_{\pi(1)}, \alpha_{\pi(2)}, \ldots, \alpha_{\pi(K)}\},
```
and the corresponding reordered dictionary atoms are:
```math
\{\mathbf{d}_{\pi(1)}, \mathbf{d}_{\pi(2)}, \ldots, \mathbf{d}_{\pi(K)}\}.
```

### Coefficient Quantization
The reordered coefficients are quantized using a Jayant quantizer. The goal is to find the minimum value of $n_\alpha$ such that
```math
MSE = \frac{1}{N} \left\| \mathbf{x} - \sum_{j=1}^{K} \widetilde{\alpha}_{\pi(j)} \mathbf{d}_{\widehat{i}_{\pi(j)}} \right\|^2,
```
where
```math
\widetilde{\alpha}_{\pi(j)} = \Delta_{\pi(j)} \left\lfloor \frac{\alpha_{\pi(j)}}{\Delta_{\pi(j)}} \right\rfloor + \frac{\Delta_{\pi(j)}}{2},
```
and
```math
\Delta_{\pi(j)} = \frac{w_{\pi(j)}}{2^{n_\alpha}}.
```
Here, $w_{\pi(j)}$ is the dynamic range of coefficient ${\pi(j)}$. For example, $w_{\pi(1)} = \sqrt{2N}$ or $w_1 = 2$ for cosine or impulse coefficients.

For subsequent coefficients
```math
w_{{\pi(j+1)}} =
\begin{cases}
\frac{w_{\pi(j)}}{2^k}, & \text{if } k > 0, \\
w_{\pi(j)}, & \text{otherwise}.
\end{cases}
```
The parameter $k$ is selected to minimize the difference between $w_{\pi(j)} / 2$ and $| \widetilde{\alpha} \_{\pi(j)} | \cdot 2^k$, subject to the constraint $w_{\pi(j)} / 2 - | \widetilde{\alpha} \_{\pi(j)} | \cdot 2^k > 0$.


### Position Encoding

The positions of the selected coefficients are encoded as follows:  
Given the set of selected basis vectors $\mathbf{d} \_{\pi(1)}, \mathbf{d} \_{\pi(2)}, \ldots, \mathbf{d} \_{\pi(K)}$, the differences between consecutive indices are computed as  
```math
\delta_k = \pi(i_{k+1}) - \pi(i_k), \quad k > 0.  
``` 
This results in the sequence  
```math
\{\pi(1), \delta_2, \ldots, \delta_K\}.  
```
Each difference $\delta_k$ is encoded using an exponential Golomb code, where one bit is used to indicate the sign.

Finally, Context-based Binary Arithmetic Coding (CABAC) is applied to improve compression efficiency. Each group of two bits—(0,0), (0,1), (1,0), and (1,1)—from the Exponential-Golomb-coded differences is encoded using distinct contexts. This approach captures the correlation between consecutive bits, further enhancing compression performance.

---

## Description of Reimplemented Method for Rate Constraint

We used the same approach as before, but for different quality levels (corresponding to a fixed number of atoms). In this case, we search for the number of atoms that minimizes the reconstruction error while ensuring that the total rate does not exceed the constraint $n_\text{tot}$.


# Prerequisites

- numpy


- matplotlib.pyplot
