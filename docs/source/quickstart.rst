Quick Start Guide
=================

This guide will help you get started with PySCSA quickly.

Basic 1D Signal Processing
---------------------------

Denoising a Simple Signal
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyscsa import SCSA1D
   import numpy as np

   # Generate a clean signal
   x = np.linspace(-10, 10, 500)
   signal = -2 * (1/np.cosh(x))**2
   
   # Add noise
   noisy = signal + 0.1 * np.random.randn(len(signal))
   
   # Apply SCSA denoising
   scsa = SCSA1D(gmma=0.5)
   result = scsa.filter_with_c_scsa(noisy)
   
   # View results
   print(f"Optimal h: {result.optimal_h:.2f}")
   print(f"PSNR: {result.metrics['psnr']:.2f} dB")
   print(f"MSE: {result.metrics['mse']:.6f}")

Manual Parameter Control
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyscsa import SCSA1D
   import numpy as np
   
   x = np.linspace(-10, 10, 200)
   signal = np.exp(-x**2)
   
   # Reconstruct with specific parameters
   scsa = SCSA1D(gmma=1.0)
   result = scsa.reconstruct(signal, h=2.0)
   
   print(f"Number of eigenvalues: {result.num_eigenvalues}")
   print(f"Reconstruction error: {result.metrics['mse']:.6f}")

Basic 2D Image Processing
--------------------------

Image Denoising
~~~~~~~~~~~~~~~

.. code-block:: python

   from pyscsa import SCSA2D
   import numpy as np
   
   # Create or load an image
   x = np.linspace(-5, 5, 100)
   y = np.linspace(-5, 5, 100)
   X, Y = np.meshgrid(x, y)
   image = np.exp(-(X**2 + Y**2) / 4)
   
   # Add noise
   noisy_image = image + 0.1 * np.random.randn(*image.shape)
   
   # Denoise using SCSA
   scsa = SCSA2D(gmma=2.0)
   denoised = scsa.denoise(
       noisy_image,
       method='windowed',
       window_size=10,
       h=5.0
   )

Using Visualization Tools
--------------------------

.. code-block:: python

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
   viz = SCSAVisualizer()
   fig = viz.plot_1d_comparison(
       signal,
       result.reconstructed,
       title="SCSA Signal Reconstruction"
   )
   plt.show()

Next Steps
----------

* Explore the :doc:`api` for detailed function references
* Check out :doc:`examples` for more advanced use cases
* Read about the :doc:`theory` behind SCSA
