# Installation

## Requirements

- Python 3.7+
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- scikit-learn >= 0.23.0
- matplotlib >= 3.2.0
- pandas >= 1.1.0

## Install via pip

```bash
pip install pyscsa
```

## Install from source

```bash
git clone https://github.com/abdou1579/pyscsa.git
cd pyscsa
pip install -e .
```

## Development installation

For contributing or running tests:

```bash
pip install -e ".[dev]"
```

This includes pytest, coverage tools, and formatting utilities.

## Verify installation

```python
import pyscsa
from pyscsa import SCSA1D
import numpy as np

x = np.linspace(-10, 10, 100)
signal = np.exp(-x**2)
scsa = SCSA1D(gmma=0.5)
result = scsa.reconstruct(signal, h=1.0)

print(f"✓ Installation successful! MSE: {result.metrics['mse']:.6f}")
```

## Troubleshooting

**Import errors**: Install missing dependencies:

```bash
pip install numpy scipy scikit-learn matplotlib pandas
```

**Version conflicts**: Use a virtual environment:

```bash
python -m venv pyscsa_env
source pyscsa_env/bin/activate  # Windows: pyscsa_env\Scripts\activate
pip install pyscsa
```