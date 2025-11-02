# Theoretical Background

## Overview

Semi-Classical Signal Analysis (SCSA) is a signal processing method inspired by quantum mechanics that interprets signals as potentials of a Schrödinger operator. Unlike Fourier or wavelet transforms with fixed bases, SCSA generates **signal-dependent** basis functions through eigenvalue decomposition.

## Mathematical Foundation

### The Schrödinger Operator

For a 1D signal \(y(x) \geq 0\), SCSA constructs:

\[
H_h(y)\psi = -h^2 \frac{d^2\psi}{dx^2} - y(x)\psi = \lambda\psi
\]

where:

- \(h > 0\): semi-classical parameter
- \(y(x)\): signal as potential
- \(\lambda\): eigenvalues
- \(\psi\): eigenfunctions in \(H^2(\mathbb{R})\)

### Signal Requirements

Must satisfy Faddeev class:

\[
y \in \mathcal{B} = \{y \in L^1_1(\mathbb{R}), \, y(x) \geq 0, \, \frac{\partial^m y}{\partial x^m} \in L^1(\mathbb{R}), \, m = 1, 2\}
\]

### Reconstruction Formula

\[
y_h(x) = 4h \sum_{n=1}^{N_h} \kappa_{nh} \psi_{nh}^2(x)
\]

where \(\lambda_{nh} = -\kappa_{nh}^2\) are negative eigenvalues with \(\kappa_{1h} > \kappa_{2h} > \cdots\)

## Semi-Classical Properties

### Convergence

As \(h \to 0\):

1. \(N_h\) increases
2. \(y_h \to y\)
3. Asymptotic: \(\lim_{h \to 0} h N_h = \frac{1}{\pi} \int_{-\infty}^{+\infty} \sqrt{y(x)} dx\)

### Eigenvalue Distribution

Eigenvalues accumulate at regular signal values, providing adaptive frequency-like decomposition based on signal structure rather than fixed sinusoids.

### Momentum Conservation

\[
\lim_{h \to 0} h \sum_{n=1}^{N_h} \kappa_{nh} = \frac{1}{4} \int_{-\infty}^{+\infty} y(x) dx
\]

\[
\lim_{h \to 0} h \sum_{n=1}^{N_h} \kappa_{nh}^3 = \frac{3}{16} \int_{-\infty}^{+\infty} y^2(x) dx
\]

## Key Parameters

### Parameter h

**Small h** (\(h \to 0\)):
- More eigenvalues
- Better approximation
- May include noise
- Higher cost

**Large h**:
- Fewer eigenvalues
- Smoother result
- Better noise suppression
- Risk of over-smoothing

**Typical**: \(h \in [0.5, 5.0]\)

### Parameter γ

In \(\gamma = 1/h^2\):

- **Small γ** (< 1): Strong smoothing
- **Large γ** (> 1): Detail preservation

### C-SCSA Automatic Selection

Optimizes \(h\) via:

\[
J = \|y_\delta - y_h\|^2_2 + \mu \int |k(t)| dt
\]

where \(k(t) = \frac{|y''_h(t)|}{(1 + y'_h(t)^2)^{3/2}}\) is curvature.

## Eigenfunction Characteristics

1. **Ordering**: Lower-order capture main features
2. **Oscillations**: \(n\)-th has \(n-1\) zeros
3. **Localization**: Concentrate around peaks
4. **Adaptivity**: Adjust to signal structure

## 2D Extension

For images, separation of variables:

\[
\mathcal{L}_{2D} = -h^2 \left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}\right) + V(x,y)
\]

Solved via:
1. 1D SCSA along rows
2. 1D SCSA along columns
3. Combine results

### Windowed Processing

For large images:
1. Divide into overlapping windows
2. Apply SCSA per window
3. Blend with weighted averaging

## Comparison

| Method | Basis | Best For |
|--------|-------|----------|
| **Fourier** | Fixed sinusoids | Periodic signals |
| **Wavelets** | Pre-selected | Multi-scale analysis |
| **SCSA** | Signal-dependent | Pulses, transients |

## Advantages

1. Adaptive basis functions
2. Excellent peak preservation
3. Physical interpretation
4. Automatic optimization
5. Robust to various noise types

## Limitations

1. \(O(N^3)\) computational cost
2. Parameter sensitivity
3. Requires \(y(x) \geq 0\)
4. Boundary effects

## Practical Guidelines

- **Smooth signals**: \(\gamma \in [0.5, 1.0]\), \(h \in [1.0, 3.0]\)
- **Noisy signals**: \(\gamma \in [0.3, 0.5]\), \(h \in [2.0, 5.0]\)
- **Detail preservation**: \(\gamma \in [1.0, 2.0]\), \(h \in [0.5, 2.0]\)

Use `filter_with_optimal_h()` for automatic selection.

## References

- Laleg-Kirati et al. (2013): "Semi-classical signal analysis"
- Li & Laleg-Kirati (2019): "Signal denoising based on Schrödinger operator"
- Helffer & Robert (1990): "Riesz means and semi-classical limits"
- Deift & Trubowitz (1979): "Inverse scattering on the line"