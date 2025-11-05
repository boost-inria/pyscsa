# Examples

## Advanced 1D Signal Processing

### Parameter Sweep Analysis

```python
from pyscsa import SCSA1D
import numpy as np
import matplotlib.pyplot as plt

# Generate signal
x = np.linspace(-10, 10, 300)
signal = np.exp(-x**2)
noisy = signal + 0.1 * np.random.randn(len(signal))

# Sweep gamma values
gammas = np.linspace(0.1, 5.0, 20)
mse_values = []
psnr_values = []

for gmma in gammas:
    scsa = SCSA1D(gmma=gmma)
    result = scsa.reconstruct(noisy, h=1.0)
    mse_values.append(result.metrics['mse'])
    psnr_values.append(result.metrics['psnr'])

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(gammas, mse_values, 'o-')
axes[0].set_xlabel('Gamma')
axes[0].set_ylabel('MSE')
axes[1].plot(gammas, psnr_values, 'o-')
axes[1].set_xlabel('Gamma')
axes[1].set_ylabel('PSNR (dB)')
plt.show()
```

### Multi-Peak Signal Reconstruction

```python
from pyscsa import SCSA1D
from pyscsa.utils import SignalGenerator, add_noise
import numpy as np
import matplotlib.pyplot as plt

# Create complex signal
x = np.linspace(-10, 10, 500)
signal = (SignalGenerator.sech_squared(x - 3) + 
          SignalGenerator.sech_squared(x + 3) + 
          0.5 * SignalGenerator.gaussian(x, sigma=1.5))

# Add noise
noisy = add_noise(signal, snr_db=10, seed=42)

# Test different gamma values
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, gmma in enumerate([0.1, 0.5, 1.0, 2.0]):
    ax = axes[idx // 2, idx % 2]
    scsa = SCSA1D(gmma=gmma)
    result = scsa.filter_with_c_scsa(noisy)
    
    ax.plot(x, signal, 'k-', label='Original', linewidth=2)
    ax.plot(x, noisy, 'gray', alpha=0.3, label='Noisy')
    ax.plot(x, result.reconstructed, 'r--', label=f'SCSA (γ={gmma})', linewidth=2)
    ax.set_title(f'gamma={gmma}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
```

### ECG Signal Denoising

```python
from pyscsa import SCSA1D
from pyscsa.visualization import SCSAVisualizer
import numpy as np

# Simulate ECG-like signal
t = np.linspace(0, 10, 1000)
ecg = (np.sin(2 * np.pi * 1.2 * t) + 
       0.5 * np.sin(2 * np.pi * 2.4 * t))

# Add noise
noisy_ecg = ecg + 0.1 * np.random.randn(len(t))

# Denoise
scsa = SCSA1D(gmma=0.3)
result = scsa.filter_with_c_scsa(noisy_ecg)

# Visualize
viz = SCSAVisualizer()
fig = viz.plot_1d_comparison(ecg, result.reconstructed, noisy_ecg, x_axis=t)
print(f"SNR improvement: {result.metrics['snr']:.2f} dB")
```

## Advanced 2D Image Processing

### Large Image Processing

```python
from pyscsa import SCSA2D
import numpy as np

# Create large image
size = 256
x = np.linspace(-10, 10, size)
y = np.linspace(-10, 10, size)
X, Y = np.meshgrid(x, y)
image = np.exp(-(X**2 + Y**2) / 10)

# Add noise
noisy_image = image + 0.1 * np.random.randn(*image.shape)

# Process with windowed approach
scsa = SCSA2D(gmma=2.0)
denoised = scsa.denoise(
    noisy_image,
    method='windowed',
    window_size=16,
    h=5.0
)
```

### Windowed Processing with Different Sizes

```python
from pyscsa import SCSA2D
import numpy as np
import matplotlib.pyplot as plt

# Create complex pattern
size = 256
x = np.linspace(-10, 10, size)
y = np.linspace(-10, 10, size)
X, Y = np.meshgrid(x, y)
image = np.exp(-(X**2 + Y**2) / 10) * np.cos(X) * np.sin(Y)
noisy_image = image + 0.05 * np.random.randn(*image.shape)

# Test different window sizes
scsa = SCSA2D(gmma=2.0)
results = {}

for ws in [8, 16, 32]:
    denoised = scsa.denoise(
        noisy_image,
        method='windowed',
        window_size=ws,
        h=5.0,
        overlap=ws//2
    )
    results[ws] = denoised

# Display
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.ravel()
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')

for idx, (ws, result) in enumerate(results.items(), 1):
    axes[idx].imshow(result, cmap='gray')
    axes[idx].set_title(f'Window size = {ws}')

plt.tight_layout()
plt.show()
```

### Comparison with Other Methods

```python
from pyscsa import SCSA2D
from pyscsa.utils import calculate_psnr, calculate_ssim
from scipy.ndimage import gaussian_filter, median_filter
import numpy as np

# Create test image
x = np.linspace(-5, 5, 128)
y = np.linspace(-5, 5, 128)
X, Y = np.meshgrid(x, y)
image = np.exp(-(X**2 + Y**2) / 4)
noisy = image + 0.05 * np.random.randn(*image.shape)

# SCSA
scsa = SCSA2D(gmma=2.0)
scsa_result = scsa.denoise(noisy, method='windowed', window_size=16, h=5.0)

# Gaussian filter
gaussian_result = gaussian_filter(noisy, sigma=1.0)

# Median filter
median_result = median_filter(noisy, size=3)

# Compare
methods = {
    'SCSA': scsa_result,
    'Gaussian': gaussian_result,
    'Median': median_result
}

print("Method Comparison:")
print("-" * 50)
for name, result in methods.items():
    psnr = calculate_psnr(image, result)
    ssim = calculate_ssim(image, result)
    print(f"{name:10s} | PSNR: {psnr:6.2f} dB | SSIM: {ssim:.4f}")
```

## Batch Processing

### Processing Multiple Signals

```python
from pyscsa import SCSA1D
import numpy as np
import pandas as pd

# Generate test signals
x = np.linspace(-10, 10, 200)
signals = {
    'exp': np.exp(-x**2),
    'sech': 1 / np.cosh(x),
    'gaussian': np.exp(-x**2 / 2),
}

# Process batch
scsa = SCSA1D(gmma=0.5)
results = []

for name, signal in signals.items():
    noisy = signal + 0.1 * np.random.randn(len(signal))
    result = scsa.filter_with_c_scsa(noisy)
    
    results.append({
        'signal': name,
        'mse': result.metrics['mse'],
        'psnr': result.metrics['psnr'],
        'optimal_h': result.optimal_h
    })

# Create DataFrame
df = pd.DataFrame(results)
print(df)
df.to_csv('batch_results.csv', index=False)
```

## Custom Signal Generation

```python
from pyscsa import SCSA1D
import numpy as np
import matplotlib.pyplot as plt

def custom_chirp(x, f0=1, f1=10):
    """Chirp signal with varying frequency"""
    t = (x - x.min()) / (x.max() - x.min())
    freq = f0 + (f1 - f0) * t
    phase = 2 * np.pi * np.cumsum(freq) / len(x)
    return np.sin(phase)

# Generate and denoise
x = np.linspace(0, 10, 1000)
signal = custom_chirp(x)
noisy = signal + 0.1 * np.random.randn(len(signal))

scsa = SCSA1D(gmma=0.5)
result = scsa.filter_with_c_scsa(noisy)

# Plot
plt.figure(figsize=(12, 4))
plt.plot(x, signal, 'k-', label='Original', linewidth=2)
plt.plot(x, noisy, 'gray', alpha=0.5, label='Noisy')
plt.plot(x, result.reconstructed, 'r--', label='SCSA', linewidth=2)
plt.legend()
plt.title('Chirp Signal Denoising')
plt.grid(True)
plt.show()
```

## Saving and Loading Results

```python
from pyscsa import SCSA1D
import numpy as np
import pickle

# Process signal
x = np.linspace(-10, 10, 200)
signal = np.exp(-x**2)

scsa = SCSA1D(gmma=0.5)
result = scsa.reconstruct(signal, h=1.0)

# Save arrays
np.save('reconstructed.npy', result.reconstructed)
np.save('eigenvalues.npy', result.eigenvalues)
np.save('eigenfunctions.npy', result.eigenfunctions)

# Save complete result
with open('scsa_result.pkl', 'wb') as f:
    pickle.dump(result, f)

# Load later
with open('scsa_result.pkl', 'rb') as f:
    loaded = pickle.load(f)

print(f"Loaded PSNR: {loaded.metrics['psnr']:.2f} dB")
```
