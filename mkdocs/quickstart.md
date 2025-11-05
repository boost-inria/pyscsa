# Quick Start Guide

## Basic 1D Signal Denoising

```python
from pyscsa import SCSA1D
import numpy as np

# Generate noisy signal
x = np.linspace(-10, 10, 500)
signal = -2 * (1/np.cosh(x))**2
noisy = signal + 0.1 * np.random.randn(len(signal))

# Denoise with automatic h optimization
scsa = SCSA1D(gmma=0.5)
result = scsa.filter_with_c_scsa(noisy)

print(f"Optimal h: {result.optimal_h:.2f}")
print(f"PSNR: {result.metrics['psnr']:.2f} dB")
print(f"MSE: {result.metrics['mse']:.6f}")
```

## Manual Parameter Control

```python
from pyscsa import SCSA1D
import numpy as np

x = np.linspace(-10, 10, 200)
signal = np.exp(-x**2)

# Reconstruct with specific h value
scsa = SCSA1D(gmma=1.0)
result = scsa.reconstruct(signal, h=2.0)

print(f"Eigenvalues: {len(result.eigenvalues)}")
print(f"MSE: {result.metrics['mse']:.6f}")
```

## 2D Image Processing

```python
from pyscsa import SCSA2D
import numpy as np

# Create test image
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)f
image = np.exp(-(X**2 + Y**2) / 4)

# Add noise
noisy_image = image + 0.1 * np.random.randn(*image.shape)

# Denoise using windowed approach
scsa = SCSA2D(gmma=2.0)
denoised = scsa.denoise(
    noisy_image,
    method='windowed',
    window_size=10,
    h=5.0
)
```

## Visualization

```python
from pyscsa import SCSA1D
from pyscsa.visualization import SCSAVisualizer
import numpy as np
import matplotlib.pyplot as plt

# Generate and process signal
x = np.linspace(-10, 10, 200)
signal = np.exp(-x**2)
noisy = signal + 0.1 * np.random.randn(len(signal))

scsa = SCSA1D(gmma=0.5)
result = scsa.filter_with_c_scsa(noisy)

# Create visualizations
viz = SCSAVisualizer(figsize=(12, 8))

# Comparison plot
fig = viz.plot_1d_comparison(
    signal,
    result.reconstructed,
    noisy,
    x_axis=x,
    metrics=result.metrics,
    title="SCSA Signal Reconstruction"
)
plt.show()

# Eigenvalue spectrum
fig = viz.plot_eigenvalues(result.eigenvalues)
plt.show()

# Eigenfunctions
fig = viz.plot_eigenfunctions(result.eigenfunctions, n_functions=6)
plt.show()
```

## Working with Real Data

```python
from pyscsa import SCSA1D
import numpy as np

# Load your data
data = np.loadtxt('signal.txt')

# Process
scsa = SCSA1D(gmma=0.5)
result = scsa.filter_with_c_scsa(data)

# Save results
np.savetxt('denoised.txt', result.reconstructed)

# Save metrics
with open('metrics.txt', 'w') as f:
    for key, value in result.metrics.items():
        f.write(f"{key}: {value}\n")
```

## Parameter Selection Tips

**For gmma (gamma)**:

- Start with `gmma=0.5` for most signals
- Use `gmma=0.1-0.3` for very noisy data
- Use `gmma=1.0-2.0` to preserve fine details

**For 2D window_size**:

- Small images (< 100×100): Use full SCSA (no windowing)
- Medium (100-500): `window_size=16-32`
- Large (> 500): `window_size=32-64`

**Overlap**: Set to `window_size // 2` for smooth blending

## Next Steps

- Read the [Theory](theory.md) to understand the mathematics
- Explore [Examples](examples.md) for advanced use cases
- Check the [API Reference](api.md) for all available functions
