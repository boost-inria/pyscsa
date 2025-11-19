# PySCSA - Semi-Classical Signal Analysis Library
[![PyPI version](https://badge.fury.io/py/pyscsa.svg)](https://badge.fury.io/py/pyscsa)
[![Documentation Status](https://readthedocs.org/projects/pyscsa/badge/?version=latest)](https://pyscsa.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/boost-inria/pyscsa/actions/workflows/tests.yml/badge.svg)](https://github.com/boost-inria/pyscsa/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for signal and image processing using Semi-Classical Signal Analysis (SCSA).

## Features

- **1D Signal Processing**: Reconstruction and denoising of 1D signals
- **2D Image Processing**: Image reconstruction using separation of variables
- **C-SCSA**: Automatic parameter optimization for filtering
- **GPU Acceleration**: CUDA-accelerated processing with CuPy (5-50x speedup)
- **Windowed Processing**: Efficient processing of large images
- **Performance Metrics**: Built-in MSE, RMSE, PSNR, and SNR calculations
- **Visualization Tools**: Comprehensive plotting utilities

## Installation

### Via pip
```bash
pip install pyscsa
```

### With GPU support
```bash
pip install pyscsa cupy-cuda11x  # For CUDA 11.x
# or
pip install pyscsa cupy-cuda12x  # For CUDA 12.x
```

### From source
```bash
git clone https://github.com/boost-inria/pyscsa.git
cd pyscsa
pip install -e .
```

## Quick Start

### 1D Signal Denoising (CPU)
```python
from pyscsa import SCSA1D
import numpy as np

# Create noisy signal
x = np.linspace(-10, 10, 500)
signal = -2 * (1/np.cosh(x))**2
noisy_signal = signal + 0.1 * np.random.randn(len(signal))

# Denoise using SCSA
scsa = SCSA1D(gmma=0.5)
result = scsa.filter_with_c_scsa(noisy_signal)

print(f"Optimal h: {result.optimal_h:.2f}")
print(f"MSE: {result.metrics['mse']:.6f}")
print(f"PSNR: {result.metrics['psnr']:.2f} dB")
```

### 1D Signal Denoising (GPU)
```python
from pyscsa import SCSA1D_GPU
import numpy as np

# Create noisy signal
x = np.linspace(-10, 10, 5000)  # Larger signal for GPU benefit
signal = -2 * (1/np.cosh(x))**2
noisy_signal = signal + 0.1 * np.random.randn(len(signal))

# Denoise using GPU-accelerated SCSA
scsa_gpu = SCSA1D_GPU(gmma=0.5, device_id=0)
result = scsa_gpu.filter_with_c_scsa(noisy_signal)

print(f"Optimal h: {result.optimal_h:.2f}")
print(f"PSNR: {result.metrics['psnr']:.2f} dB")
```

### 2D Image Reconstruction (CPU)
```python
from pyscsa import SCSA2D
import numpy as np

# Load or create image
image = np.random.rand(100, 100)

# Add noise
noisy_image = image + 0.05 * np.random.randn(*image.shape)

# Denoise using windowed SCSA
scsa = SCSA2D(gmma=2.0)
denoised = scsa.denoise(noisy_image, method='windowed', window_size=8, h=5.0)
```

### 2D Image Reconstruction (GPU)
```python
from pyscsa import SCSA2D_GPU
import numpy as np

# Load or create image
image = np.random.rand(256, 256)
noisy_image = image + 0.05 * np.random.randn(*image.shape)

# GPU-accelerated reconstruction
scsa_gpu = SCSA2D_GPU(gmma=2.0, device_id=0)
result = scsa_gpu.reconstruct(noisy_image, h=10.0)

denoised = result.reconstructed
print(f"PSNR: {result.metrics['psnr']:.2f} dB")
```

## GPU Acceleration

PySCSA supports GPU acceleration through CuPy, providing significant speedup for large signals and images:

**Requirements:**
- NVIDIA GPU with CUDA support
- CuPy installed (`pip install cupy-cuda11x` or `cupy-cuda12x`)

**Check GPU availability:**
```python
from pyscsa import CUPY_AVAILABLE
print(f"GPU acceleration available: {CUPY_AVAILABLE}")
```

## Running Tests

```bash
python -m pytest tests/ -v
```

With coverage report:
```bash
python -m pytest tests/ --cov=pyscsa --cov-report=html
```

## Documentation

Full documentation is available at [https://pyscsa.readthedocs.io](https://pyscsa.readthedocs.io)

## API Reference

### CPU Classes
- `SCSA1D`: 1D signal processing
- `SCSA2D`: 2D image processing
- `SCSAVisualizer`: Visualization utilities

### GPU Classes
- `SCSA1D_GPU`: GPU-accelerated 1D processing
- `SCSA2D_GPU`: GPU-accelerated 2D processing

All GPU classes maintain the same API as their CPU counterparts with an additional `device_id` parameter for multi-GPU systems.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{pyscsa,
  title = {PySCSA: Python Semi-Classical Signal Analysis Library},
  author = {boost inria},
  year = {2025},
  url = {https://github.com/boost-inria/pyscsa}
}
```

## License

This project is licensed under the Inria License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the Semi-Classical Signal Analysis method
