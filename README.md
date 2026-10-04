# PySCSA — Semi-Classical Signal Analysis

[![PyPI version](https://badge.fury.io/py/pyscsa.svg)](https://badge.fury.io/py/pyscsa)
[![Documentation](https://readthedocs.org/projects/pyscsa/badge/?version=latest)](https://pyscsa.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/boost-inria/pyscsa/actions/workflows/tests.yml/badge.svg)](https://github.com/boost-inria/pyscsa/actions)

A Python library for signal and image processing based on the Semi-Classical Signal Analysis (SCSA) framework.

---

## How It Works

SCSA treats a signal $y(x) \geq 0$ as the potential of a Schrödinger operator $\mathcal{H}_h = -h^2 d^2/dx^2 - y(x)$. Its negative eigenvalues $\{-\kappa_n^2\}$ and $L^2$-normalized eigenfunctions $\{\psi_n\}$ encode the signal's structure, and the signal is reconstructed as:

$$y_h(x) = 4h \sum_{n=1}^{N_h} \kappa_n\, \psi_n^2(x)$$

The atoms $\phi_n = \psi_n^2$ are **non-negative**, **spatially localized**, and **signal-adaptive** — their shape is determined by $y$ itself with no fixed dictionary. The parameter $h$ controls resolution: smaller $h$ yields more atoms and finer detail; larger $h$ retains only dominant structures and suppresses noise.

**C-SCSA** automates the selection of $h$ by minimizing a data-driven cost balancing reconstruction fidelity against the geometric curvature of $y_h$ — a proxy for noise energy (Li & Laleg-Kirati, IET Signal Processing 2021). No knowledge of peak locations or noise level is required.

For the full mathematical treatment see the [documentation](https://pyscsa.readthedocs.io) and references below.

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
- **C-SCSA** — automatic bandwidth parameter selection via curvature penalty, no peak localization required
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
  author = {A. Guir, I.J.S. Filho, J.M. Vargas and T.M. Laleg},
  year   = {2025},
  url    = {https://github.com/boost-inria/pyscsa}
}
```

```bibtex
@article{li2021cscsa,
  title   = {Signal denoising based on the Schrödinger operator's eigenspectrum and a curvature constraint},
  author  = {Li, P. and Laleg-Kirati, T.M.},
  journal = {IET Signal Processing},
  volume  = {15},
  pages   = {195--206},
  year    = {2021},
  doi     = {10.1049/sil2.12023}
}
```

```bibtex
@misc{lalegkirati2010scsa,
  title         = {Semi-classical signal analysis},
  author        = {Laleg-Kirati Taous-Meriem, Crépeau Emmanuelle and Sorine Michel},
  year          = {2010},
  eprint        = {1007.0938},
  archivePrefix = {arXiv},
  primaryClass  = {math-ph}
}
```

---

## License

Inria License — see [LICENSE](LICENSE) for details.
