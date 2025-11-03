# PySCSA - Semi-Classical Signal Analysis

[![Tests](https://img.shields.io/badge/tests-47%20passed-brightgreen)](https://github.com/boost-inria/pyscsa)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PySCSA** is a Python library for signal and image processing using Semi-Classical Signal Analysis (SCSA), a method inspired by quantum mechanics.

## Key Features

- **Adaptive Signal Decomposition**: Signal-dependent basis functions that adapt to your data
- **1D Signal Processing**: Reconstruction and denoising of 1D signals
- **2D Image Processing**: Image reconstruction using separation of variables
- **Automatic Optimization**: C-SCSA for parameter optimization
- **Windowed Processing**: Efficient handling of large images
- **Comprehensive Metrics**: Built-in MSE, PSNR, SSIM calculations

## Quick Example

```python
from pyscsa import SCSA1D
import numpy as np

# Generate noisy signal
x = np.linspace(-10, 10, 500)
signal = -2 * (1/np.cosh(x))**2
noisy = signal + 0.1 * np.random.randn(len(signal))

# Denoise using SCSA
scsa = SCSA1D(gmma=0.5)
result = scsa.filter_with_optimal_h(np.abs(noisy))

print(f"PSNR: {result.metrics['psnr']:.2f} dB")
```

## Why SCSA?

Unlike traditional methods that use fixed basis functions (like Fourier's sines/cosines or predefined wavelets), SCSA generates **signal-dependent** basis functions through eigenvalue decomposition of the Schrödinger operator. This provides:

- **Better Peak Preservation**: Ideal for pulse-shaped signals
- **Adaptive Resolution**: Automatically adjusts to signal features
- **Quantum-Inspired**: Solid mathematical foundation from quantum mechanics
- **Effective Denoising**: Separates signal from noise naturally

## Applications

- Biomedical signal processing (ECG, arterial blood pressure)
- Magnetic Resonance Spectroscopy (MRS)
- Image denoising and enhancement
- Feature extraction for machine learning

## Installation

```bash
pip install pyscsa
```

Or from source:

```bash
git clone https://github.com/boost-inria/pyscsa.git
cd pyscsa
pip install -e .
```

## Get Started

Check out our [Quick Start Guide](quickstart.md) or dive into the [API Reference](api.md).

## Citation

If you use PySCSA in your research:

```bibtex
@software{pyscsa,
  title = {PySCSA: Python Semi-Classical Signal Analysis Library},
  author = {boost inria},
  year = {2025},
  url = {https://github.com/boost-inria/pyscsa}
}
```

## License

MIT License - see [LICENSE](https://github.com/boost-inria/pyscsa/blob/main/LICENSE) for details.
