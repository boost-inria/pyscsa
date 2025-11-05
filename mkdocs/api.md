# API Reference

Complete reference for PySCSA classes and functions.

---

## Core Classes

### SCSA1D

1D Semi-Classical Signal Analysis for signal reconstruction and filtering.

```python
from pyscsa import SCSA1D

scsa = SCSA1D(gmma=0.5)
```

**Parameters:**

- `gmma` (float, default=0.5): Gamma parameter controlling smoothness

**Methods:**

#### `reconstruct(signal, h=1.0, lambda_g=None)`

Reconstruct 1D signal using SCSA.

**Parameters:**

- `signal` (ndarray): Input signal (auto-converted to positive)
- `h` (float, default=1.0): Semi-classical parameter
- `lambda_g` (float, optional): Lambda threshold (default=0)

**Returns:**

- `SCSAResult`: Contains:
  - `reconstructed` (ndarray): Reconstructed signal
  - `eigenvalues` (ndarray): Eigenvalues used
  - `eigenfunctions` (ndarray): Eigenfunctions matrix
  - `num_eigenvalues` (int): Number of eigenvalues
  - `metrics` (dict): MSE, RMSE, PSNR, SNR

**Example:**

```python
result = scsa.reconstruct(signal, h=1.0)
print(f"MSE: {result.metrics['mse']:.6f}")
print(f"PSNR: {result.metrics['psnr']:.2f} dB")
```

---

#### `filter_with_c_scsa(signal, curvature_weight=4.0, h_range=None)`

Filter signal using C-SCSA with automatic h optimization.

**Parameters:**

- `signal` (ndarray): Input noisy signal
- `curvature_weight` (float, default=4.0): Weight for curvature penalty
- `h_range` (tuple, optional): (h_min, h_max) for search range

**Returns:**

- `SCSAResult`: Result with `optimal_h` attribute set

**Example:**

```python
result = scsa.filter_with_c_scsa(noisy_signal)
print(f"Optimal h: {result.optimal_h:.2f}")
print(f"PSNR: {result.metrics['psnr']:.2f} dB")
```

---

#### `denoise(noisy_signal, **kwargs)`

Convenience method for signal denoising.

**Parameters:**

- `noisy_signal` (ndarray): Input noisy signal
- `**kwargs`: Passed to `filter_with_c_scsa()`

**Returns:**

- `ndarray`: Denoised signal (just the array, not SCSAResult)

**Example:**

```python
denoised = scsa.denoise(noisy_signal)
```

---

### SCSA2D

2D Semi-Classical Signal Analysis for image reconstruction.

```python
from pyscsa import SCSA2D

scsa = SCSA2D(gmma=2.0)
```

**Parameters:**

- `gmma` (float, default=2.0): Gamma parameter for 2D processing

**Methods:**

#### `reconstruct(image, h=10.0, lambda_g=0)`

Reconstruct 2D image using separation of variables.

**Parameters:**

- `image` (ndarray): Input 2D image
- `h` (float, default=10.0): Semi-classical parameter
- `lambda_g` (float, default=0): Lambda threshold

**Returns:**

- `SCSAResult`: Contains reconstructed image and metrics

**Example:**

```python
result = scsa.reconstruct(image, h=5.0)
reconstructed_image = result.reconstructed
```

---

#### `reconstruct_windowed(image, h=1.0, window_size=4, stride=1, lambda_g=0)`

Reconstruct using windowed approach for large images.

**Parameters:**

- `image` (ndarray): Input 2D image
- `h` (float, default=10.0): Semi-classical parameter
- `window_size` (int, default=4): Size of sliding window
- `stride` (int, default=1): Stride for window
- `lambda_g` (float, default=0): Lambda threshold

**Returns:**

- `ndarray`: Reconstructed image

**Example:**

```python
result = scsa.reconstruct_windowed(
    large_image, 
    h=1.0, 
    window_size=16, 
    stride=8
)
```

---

#### `denoise(noisy_image, method='windowed', **kwargs)`

Denoise 2D image.

**Parameters:**

- `noisy_image` (ndarray): Input noisy image
- `method` (str, default='windowed'): 'standard' or 'windowed'
- `**kwargs`: Passed to reconstruction method

**Returns:**

- `ndarray`: Denoised image

**Example:**

```python
denoised = scsa.denoise(
    noisy_image,
    method='windowed',
    window_size=16,
    h=5.0
)
```

---

## Visualization

### SCSAVisualizer

Visualization utilities for SCSA results.

```python
from pyscsa.visualization import SCSAVisualizer

viz = SCSAVisualizer(figsize=(12, 8), style='default')
```

**Parameters:**

- `figsize` (tuple, default=(10, 6)): Figure size (width, height)
- `style` (str, default='default'): Matplotlib style

**Methods:**

#### `plot_1d_comparison(original, reconstructed, noisy=None, title='', x_axis=None, metrics=None)`

Plot 1D signal comparison.

**Parameters:**

- `original` (ndarray): Original signal
- `reconstructed` (ndarray): Reconstructed signal
- `noisy` (ndarray, optional): Noisy signal
- `title` (str): Plot title
- `x_axis` (ndarray, optional): X-axis values
- `metrics` (dict, optional): Metrics to display

**Returns:**

- `Figure`: Matplotlib figure

**Example:**

```python
fig = viz.plot_1d_comparison(
    signal, 
    result.reconstructed, 
    noisy,
    title="SCSA Denoising",
    metrics=result.metrics
)
```

---

#### `plot_2d_comparison(original, reconstructed, noisy=None, title='', cmap='gray', metrics=None)`

Plot 2D image comparison.

**Parameters:**

- `original` (ndarray): Original image
- `reconstructed` (ndarray): Reconstructed image
- `noisy` (ndarray, optional): Noisy image
- `title` (str): Plot title
- `cmap` (str, default='gray'): Colormap
- `metrics` (dict, optional): Metrics to display

**Returns:**

- `Figure`: Matplotlib figure

---

#### `plot_eigenvalues(eigenvalues, title='Eigenvalue Spectrum')`

Plot eigenvalue spectrum.

**Parameters:**

- `eigenvalues` (ndarray): Eigenvalues array
- `title` (str): Plot title

**Returns:**

- `Figure`: Matplotlib figure with linear and log scale

---

#### `plot_eigenfunctions(eigenfunctions, n_functions=6, title='Eigenfunctions')`

Plot eigenfunctions.

**Parameters:**

- `eigenfunctions` (ndarray): Eigenfunctions matrix
- `n_functions` (int, default=6): Number to plot
- `title` (str): Plot title

**Returns:**

- `Figure`: Matplotlib figure

---

#### `plot_parameter_sweep(param_values, metrics, param_name='parameter', optimal_value=None)`

Plot parameter sweep results.

**Parameters:**

- `param_values` (ndarray): Parameter values tested
- `metrics` (dict): Dict of metric_name: values
- `param_name` (str): Parameter name for label
- `optimal_value` (float, optional): Mark optimal value

**Returns:**

- `Figure`: Matplotlib figure

---

#### `save_figure(fig, filename, dpi=300, format='png')`

Save figure to file.

**Parameters:**

- `fig` (Figure): Matplotlib figure
- `filename` (str): Output filename
- `dpi` (int, default=300): Resolution
- `format` (str, default='png'): File format

---

## Utility Functions

### Signal Generation

```python
from pyscsa.utils import SignalGenerator
```

**Methods:**

- `SignalGenerator.sech_squared(x)`: Generate sech² signal
- `SignalGenerator.gaussian(x, sigma=1.0)`: Gaussian signal
- `SignalGenerator.double_well(x, separation=3, depth=50)`: Double-well potential

---

### Noise and Metrics

```python
from pyscsa.utils import add_noise, calculate_mse, calculate_psnr
```

#### `add_noise(signal, snr_db, seed=None)`

Add Gaussian noise to signal.

**Parameters:**

- `signal` (ndarray): Input signal
- `snr_db` (float): Desired SNR in dB
- `seed` (int, optional): Random seed

**Returns:**

- `ndarray`: Noisy signal

---

#### `calculate_mse(original, reconstructed)`

Calculate Mean Squared Error.

**Returns:** float

---

#### `calculate_psnr(original, reconstructed)`

Calculate Peak Signal-to-Noise Ratio.

**Returns:** float (dB)

---

#### `calculate_ssim(original, reconstructed)`

Calculate Structural Similarity Index.

**Returns:** float [0, 1]

---

## Data Classes

### SCSAResult

Result container returned by SCSA methods.

**Attributes:**

- `reconstructed` (ndarray): Reconstructed signal/image
- `eigenvalues` (ndarray): Eigenvalues used
- `eigenfunctions` (ndarray): Eigenfunctions matrix
- `num_eigenvalues` (int): Number of eigenvalues
- `optimal_h` (float, optional): Optimal h value (if computed)
- `metrics` (dict): Performance metrics

**Example:**

```python
result = scsa.reconstruct(signal, h=1.0)
print(result.reconstructed.shape)
print(result.num_eigenvalues)
print(result.metrics)
```

---

## Complete Example

```python
from pyscsa import SCSA1D, SCSA2D
from pyscsa.visualization import SCSAVisualizer
from pyscsa.utils import add_noise
import numpy as np
import matplotlib.pyplot as plt

# 1D Example
x = np.linspace(-10, 10, 500)
signal = np.exp(-x**2)
noisy = add_noise(signal, snr_db=15, seed=42)

scsa1d = SCSA1D(gmma=0.5)
result = scsa1d.filter_with_c_scsa(noisy)

viz = SCSAVisualizer()
fig = viz.plot_1d_comparison(signal, result.reconstructed, noisy)
plt.show()

# 2D Example
image = np.random.rand(100, 100)
noisy_image = add_noise(image, snr_db=20, seed=42)

scsa2d = SCSA2D(gmma=2.0)
denoised = scsa2d.denoise(
    noisy_image, 
    method='windowed',
    window_size=16, 
    h=5.0
)

fig = viz.plot_2d_comparison(image, denoised, noisy_image)
plt.show()
```

---

For more details, see [ReadTheDocs](https://pyscsa.readthedocs.io)
