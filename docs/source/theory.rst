Theoretical Background
======================

Overview
--------

Semi-Classical Signal Analysis (SCSA) is a signal processing method inspired by 
quantum mechanics that interprets signals as potentials of a Schrödinger operator. 
Unlike traditional methods like Fourier or wavelet transforms that use fixed basis 
functions, SCSA generates signal-dependent basis functions through eigenvalue 
decomposition, allowing adaptive representation that captures the signal's 
morphological characteristics.

Mathematical Foundation
-----------------------

The Schrödinger Operator
~~~~~~~~~~~~~~~~~~~~~~~~~

For a 1D signal :math:`y(x) \geq 0`, SCSA constructs the semi-classical Schrödinger operator:

.. math::

   H_h(y)\psi = -h^2 \frac{d^2\psi}{dx^2} - y(x)\psi = \lambda\psi, \quad \psi \in H^2(\mathbb{R})

where:

* :math:`h > 0` is the semi-classical parameter (Planck's constant analog)
* :math:`y(x)` is the signal interpreted as the potential
* :math:`\lambda` are the eigenvalues
* :math:`\psi` are the eigenfunctions in the Sobolev space :math:`H^2(\mathbb{R})`

Signal Admissibility Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The signal must satisfy the Faddeev class conditions:

.. math::

   y \in \mathcal{B} = \{y \in L^1_1(\mathbb{R}), \, y(x) \geq 0, \, \forall x \in \mathbb{R}, \, 
   \frac{\partial^m y}{\partial x^m} \in L^1(\mathbb{R}), \, m = 1, 2\}

where :math:`L^1_1(\mathbb{R}) = \{V \mid \int_{-\infty}^{+\infty} |V(x)|(1 + |x|)dx < \infty\}`.

Signal Reconstruction Formula
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The signal is reconstructed using the discrete spectrum:

.. math::

   y_h(x) = 4h \sum_{n=1}^{N_h} \kappa_{nh} \psi_{nh}^2(x)

where:

* :math:`\lambda_{nh} = -\kappa_{nh}^2` are the negative eigenvalues with :math:`\kappa_{1h} > \kappa_{2h} > \cdots > \kappa_{N_h h}`
* :math:`\psi_{nh}` are the :math:`L^2`-normalized eigenfunctions
* :math:`N_h` is the number of negative eigenvalues (depends on :math:`h`)

Semi-Classical Properties
--------------------------

Convergence as h → 0
~~~~~~~~~~~~~~~~~~~~

As the semi-classical parameter :math:`h` decreases:

1. The number of negative eigenvalues :math:`N_h` increases (Proposition 2.1)
2. The reconstruction :math:`y_h` converges to the original signal :math:`y`
3. The asymptotic behavior follows:

.. math::

   \lim_{h \to 0} h N_h = \frac{1}{\pi} \int_{-\infty}^{+\infty} \sqrt{y(x)} dx

Eigenvalue Distribution
~~~~~~~~~~~~~~~~~~~~~~~

The eigenvalues provide a natural quantization of the signal. For regular values of :math:`y`, 
the eigenvalues :math:`\kappa_{nh}^2` accumulate at these regular values, providing an 
adaptive frequency-like decomposition that depends on the signal's structure rather than 
fixed sinusoidal bases.

Key Parameters
--------------

The Semi-Classical Parameter h
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter :math:`h` controls the trade-off between signal fidelity and smoothness:

* **Small h** (:math:`h \to 0`): 
  
  - More eigenvalues included (:math:`N_h` increases)
  - Better approximation of the original signal
  - May include noise components
  - Higher computational cost

* **Large h**: 
  
  - Fewer eigenvalues (:math:`N_h` decreases)
  - Smoother reconstruction
  - Better noise suppression
  - Risk of over-smoothing

**Typical values**:

- Smooth signals: :math:`h \in [0.5, 2.0]`
- Noisy signals: :math:`h \in [1.0, 5.0]`
- Detailed preservation: :math:`h \in [0.1, 1.0]`

The Gamma Parameter γ
~~~~~~~~~~~~~~~~~~~~~

In the notation :math:`\gamma = 1/h^2 = \chi`, gamma controls the eigenvalue spacing:

* **Small γ** (γ < 1): Equivalent to large h, provides strong smoothing
* **Large γ** (γ > 1): Equivalent to small h, preserves fine details

The Cutoff Parameter
~~~~~~~~~~~~~~~~~~~~

For denoising applications, eigenvalues can be thresholded below a cutoff :math:`\lambda_g`:

.. math::

   y_h(x) = 4h \sum_{\kappa_{nh}^2 < \lambda_g} \kappa_{nh} \psi_{nh}^2(x)

Automatic Parameter Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The C-SCSA method automatically optimizes :math:`h` using a cost function with 
curvature constraints:

.. math::

   J = ||y_\delta - y_h||^2_2 + \mu \int |k(t)| dt

where :math:`k(t) = \frac{|y''_h(t)|}{(1 + y'_h(t)^2)^{3/2}}` is the curvature, 
balancing fidelity to data and smoothness.

Eigenfunction Characteristics
------------------------------

The eigenfunctions :math:`\psi_{nh}` have special properties:

1. **Ordering**: Lower-order eigenfunctions (small :math:`n`) correspond to 
   large eigenvalues and capture main signal features

2. **Oscillations**: The :math:`n`-th eigenfunction has :math:`n-1` zeros, with 
   higher-order functions being more oscillatory

3. **Localization**: Eigenfunctions naturally localize around signal peaks and 
   features

4. **Adaptivity**: Unlike Fourier or wavelet bases, eigenfunctions adapt to the 
   specific signal structure

Connection to Inverse Spectral Theory
--------------------------------------

SCSA is related to inverse spectral problems for Schrödinger operators. The 
Deift-Trubowitz trace formula provides the connection:

.. math::

   y(x) = -4 \sum_{n=1}^{N} \kappa_n \psi_n^2(x) + \frac{2i}{\pi} \int_{-\infty}^{+\infty} k R(k) f_\pm^2(k,x) dk

For reflectionless potentials (when reflection coefficient :math:`R(k) = 0`), the 
reconstruction is exact using only the discrete spectrum, establishing the 
theoretical foundation for SCSA's effectiveness with pulse-shaped signals.

Advantages Over Classical Methods
----------------------------------

**Versus Fourier Transform**:

- Adaptive basis vs. fixed sinusoids
- Better for non-periodic, localized signals
- Natural handling of pulse-shaped signals

**Versus Wavelets**:

- Signal-dependent basis vs. pre-selected wavelets
- Automatic scale selection through eigenvalues
- Direct physical interpretation via quantum mechanics

**Versus Empirical Mode Decomposition**:

- Rigorous mathematical foundation
- Well-defined convergence properties
- Systematic parameter selection methods

Computational Considerations
-----------------------------

The numerical implementation uses Fourier pseudo-spectral methods to discretize 
the differential operator, yielding a matrix eigenvalue problem:

.. math::

   (-h^2 D_2 - \text{diag}(Y)) \psi = \lambda \psi

where :math:`D_2` is the second-order differentiation matrix and complexity is 
:math:`O(N^3)` for :math:`N` sample points.

References
----------

For detailed mathematical proofs and applications:

* Laleg-Kirati et al. (2010): "Semi-classical signal analysis" - Original SCSA paper
* Li & Laleg-Kirati (2019): "Signal denoising based on the Schrödinger operator's eigenspectrum"
* Helffer & Robert (1990): Riesz means and semi-classical limits
* Deift & Trubowitz (1979): Inverse scattering on the line
