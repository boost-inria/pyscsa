# PySCSA — Semi-Classical Signal Analysis

[![PyPI version](https://badge.fury.io/py/pyscsa.svg)](https://badge.fury.io/py/pyscsa)
[![Documentation](https://readthedocs.org/projects/pyscsa/badge/?version=latest)](https://pyscsa.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/boost-inria/pyscsa/actions/workflows/tests.yml/badge.svg)](https://github.com/boost-inria/pyscsa/actions)

A Python library for signal and image processing based on the Semi-Classical Signal Analysis (SCSA) framework.

---

## Installation

```bash
pip install pyscsa
```

For development:
```bash
git clone https://github.com/boost-inria/pyscsa.git
cd pyscsa
pip install -e .
```

---

## Quick Start

### 1D Signal Denoising

```python
from pyscsa import SCSA1D
import numpy as np

x = np.linspace(-10, 10, 500)
signal = -2 / np.cosh(x)**2
noisy = signal + 0.1 * np.random.randn(len(signal))

scsa = SCSA1D(gmma=0.5)
result = scsa.filter_with_c_scsa(noisy)

print(f"Optimal h : {result.optimal_h:.4f}")
print(f"PSNR      : {result.metrics['psnr']:.2f} dB")
```

### 2D Image Denoising

```python
from pyscsa import SCSA2D
import numpy as np

image = np.random.rand(128, 128)
noisy = image + 0.05 * np.random.randn(*image.shape)

scsa = SCSA2D(gmma=2.0)
denoised = scsa.denoise(noisy, method='windowed', window_size=8, h=5.0)
```

---

## Features

- **1D & 2D processing** — reconstruction and denoising via spectral decomposition of a Schrödinger-type operator
- **C-SCSA** — automatic bandwidth parameter selection
- **Windowed mode** — memory-efficient processing of large images
- **Performance metrics** — MSE, RMSE, PSNR, SNR out of the box
- **Visualization** — built-in plotting utilities via `SCSAVisualizer`

---

## API Reference

| Class | Description |
|---|---|
| `SCSA1D` | 1D signal reconstruction and denoising |
| `SCSA2D` | 2D image reconstruction (full and windowed) |
| `SCSAVisualizer` | Plotting and diagnostic tools |

Full documentation: [pyscsa.readthedocs.io](https://pyscsa.readthedocs.io)

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=pyscsa --cov-report=html  # with coverage
```

---

## Citation

```bibtex
@software{pyscsa,
  title  = {PySCSA: Python Semi-Classical Signal Analysis Library},
  author = {A. Guir and I.J.S. Filho and J.M. Vargas and T.M. Laleg},
  year   = {2025},
  url    = {https://github.com/boost-inria/pyscsa}
}
```

```bibtex
@misc{lalegkirati2010scsa,
  title         = {Semi-classical signal analysis},
  author        = {Laleg-Kirati, Taous-Meriem and Crépeau, Emmanuelle and Sorine, Michel},
  year          = {2010},
  eprint        = {1007.0938},
  archivePrefix = {arXiv},
  primaryClass  = {math-ph}
}
```

---

## License

Inria License — see [LICENSE](LICENSE) for details.
