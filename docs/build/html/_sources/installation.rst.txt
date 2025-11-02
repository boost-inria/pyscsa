Installation
============

Requirements
------------

PySCSA requires Python 3.7 or later and the following packages:

* numpy >= 1.19.0
* scipy >= 1.5.0
* scikit-learn >= 0.23.0
* matplotlib >= 3.2.0
* pandas >= 1.1.0

Install from PyPI
-----------------

The easiest way to install PySCSA is using pip:

.. code-block:: bash

   pip install pyscsa

Install from Source
-------------------

To install the latest development version:

.. code-block:: bash

   git clone https://github.com/abdou1579/pyscsa.git
   cd pyscsa
   pip install -e .

Development Installation
------------------------

If you want to contribute or run tests:

.. code-block:: bash

   git clone https://github.com/abdou1579/pyscsa.git
   cd pyscsa
   pip install -e ".[dev]"

This installs additional development dependencies including pytest, coverage tools, 
and code formatting utilities.

Verify Installation
-------------------

To verify your installation:

.. code-block:: python

   import pyscsa
   from pyscsa import SCSA1D
   import numpy as np
   
   x = np.linspace(-10, 10, 100)
   signal = np.exp(-x**2)
   scsa = SCSA1D(gmma=0.5)
   result = scsa.reconstruct(signal, h=1.0)
   print(f"Reconstruction successful! MSE: {result.metrics['mse']:.6f}")

Troubleshooting
---------------

**Import errors**: Make sure all dependencies are installed:

.. code-block:: bash

   pip install numpy scipy scikit-learn matplotlib pandas

**Version conflicts**: Create a fresh virtual environment:

.. code-block:: bash

   python -m venv pyscsa_env
   source pyscsa_env/bin/activate  # On Windows: pyscsa_env\Scripts\activate
   pip install pyscsa