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
- **Windowed Processing**: Efficient processing of large images
- **Performance Metrics**: Built-in MSE, RMSE, PSNR, and SNR calculations
- **Visualization Tools**: Comprehensive plotting utilities

## Installation

### Via pip

```bash
pip install pyscsa
```

### From source

```bash
git clone https://github.com/boost-inria/pyscsa.git
cd pyscsa
pip install -e .
```

## Quick Start

### 1D Signal Denoising

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

### 2D Image reconstruction

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the Semi-Classical Signal Analysis method
- Inspired by quantum mechanical approaches to signal processing
