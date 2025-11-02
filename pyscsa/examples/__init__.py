"""
SCSA Examples Module
====================

This module provides example implementations and demonstrations of SCSA functionality.

Available Examples:
-------------------
Signal Processing:
    - basic_1d_denoising: Simple 1D signal denoising
    - adaptive_filtering: Adaptive SCSA filtering demonstration
    - robust_denoising: Robust SCSA with outlier handling
    - parameter_optimization: Finding optimal parameters

Image Processing:
    - basic_2d_reconstruction: Simple 2D image reconstruction
    - windowed_processing: Windowed SCSA for large images
    - multiscale_reconstruction: Multi-scale image processing

Utilities:
    - generate_test_signals: Create various test signals
    - generate_test_images: Create various test images
    - run_all_examples: Execute all examples
    - quick_demo: Quick demonstration of SCSA capabilities
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
import matplotlib.pyplot as plt

# Import main library components
from pyscsa import SCSA1D, SCSA2D, add_noise
from pyscsa.utils import SignalGenerator, NoiseGenerator
from pyscsa.filters import AdaptiveSCSA, RobustSCSA, MultiScaleSCSA
from pyscsa.visualization import SCSAVisualizer
from pyscsa.metrics import QualityMetrics


# Example functions that can be imported and used directly
def basic_1d_denoising(signal: Optional[np.ndarray] = None, 
                       snr_db: float = 20,
                       gamma: float = 0.5,
                       plot: bool = True) -> Dict[str, Any]:
    """
    Basic 1D signal denoising example.
    
    Parameters
    ----------
    signal : np.ndarray, optional
        Input signal. If None, generates a test signal
    snr_db : float
        Signal-to-noise ratio in dB
    gamma : float
        Gamma parameter for SCSA
    plot : bool
        Whether to display plots
        
    Returns
    -------
    dict
        Results dictionary containing denoised signal and metrics
    """
    if signal is None:
        # Generate default test signal
        x = np.linspace(-10, 10, 500)
        signal = SignalGenerator.sech_squared(x, center=0, width=1)
    
    # Add noise
    noisy = add_noise(signal, snr_db=snr_db, seed=42)
    
    # Apply SCSA
    scsa = SCSA1D(gamma=gamma)
    result = scsa.filter_with_optimal_h(np.abs(noisy))
    
    # Compute metrics
    metrics = QualityMetrics.compute_all(np.abs(signal), result.reconstructed)
    
    # Plot if requested
    if plot:
        viz = SCSAVisualizer()
        fig = viz.plot_1d_comparison(
            np.abs(signal), 
            result.reconstructed,
            np.abs(noisy),
            title=f"1D Denoising (SNR={snr_db}dB, γ={gamma})"
        )
        plt.show()
    
    return {
        'original': signal,
        'noisy': noisy,
        'denoised': result.reconstructed,
        'optimal_h': result.optimal_h,
        'metrics': {m.name: m.value for m in metrics.values()}
    }


def basic_2d_reconstruction(image: Optional[np.ndarray] = None,
                           snr_db: float = 15,
                           gamma: float = 2.0,
                           h: float = 10.0,
                           method: str = 'windowed',
                           plot: bool = True) -> Dict[str, Any]:
    """
    Basic 2D image reconstruction example.
    
    Parameters
    ----------
    image : np.ndarray, optional
        Input image. If None, generates a test image
    snr_db : float
        Signal-to-noise ratio in dB
    gamma : float
        Gamma parameter for SCSA
    h : float
        h parameter
    method : str
        Processing method ('standard' or 'windowed')
    plot : bool
        Whether to display plots
        
    Returns
    -------
    dict
        Results dictionary containing denoised image and metrics
    """
    if image is None:
        # Generate default test image
        image = generate_test_image(size=100, pattern='gaussian')
    
    # Add noise
    noisy = add_noise(image, snr_db=snr_db, seed=42)
    
    # Apply SCSA
    scsa = SCSA2D(gamma=gamma)
    denoised = scsa.denoise(np.abs(noisy), method=method, h=h, window_size=8)
    
    # Compute metrics
    metrics = QualityMetrics.compute_all(image, denoised)
    
    # Plot if requested
    if plot:
        viz = SCSAVisualizer()
        fig = viz.plot_2d_comparison(
            image,
            denoised,
            noisy,
            title=f"2D Reconstruction ({method}, SNR={snr_db}dB)"
        )
        plt.show()
    
    return {
        'original': image,
        'noisy': noisy,
        'denoised': denoised,
        'metrics': {m.name: m.value for m in metrics.values()}
    }


def adaptive_filtering(signal: Optional[np.ndarray] = None,
                      snr_db: float = 18,
                      plot: bool = True) -> Tuple[Dict, Dict]:
    """
    Compare standard vs adaptive SCSA filtering.
    
    Parameters
    ----------
    signal : np.ndarray, optional
        Input signal. If None, generates a complex test signal
    snr_db : float
        Signal-to-noise ratio in dB
    plot : bool
        Whether to display plots
        
    Returns
    -------
    tuple
        (standard_results, adaptive_results) dictionaries
    """
    if signal is None:
        # Generate complex signal with varying characteristics
        x = np.linspace(0, 30, 1500)
        signal = np.zeros_like(x)
        signal[x < 10] = 5 * np.exp(-0.3 * x[x < 10])
        signal[(x >= 10) & (x < 15)] = 2
        signal[(x >= 15) & (x < 20)] = 2 + 3 * np.sin(4 * np.pi * x[(x >= 15) & (x < 20)])
        signal[x >= 20] = 3
    else:
        x = np.arange(len(signal))
    
    # Add noise
    noisy = add_noise(signal, snr_db=snr_db, seed=42)
    
    # Standard SCSA
    scsa_standard = SCSA1D(gamma=0.5)
    result_standard = scsa_standard.filter_with_optimal_h(np.abs(noisy))
    
    # Adaptive SCSA
    scsa_adaptive = AdaptiveSCSA(base_gamma=0.5)
    result_adaptive = scsa_adaptive.denoise(np.abs(noisy), adapt_h=True, adapt_gamma=True)
    
    # Prepare results
    standard_results = {
        'denoised': result_standard.reconstructed,
        'optimal_h': result_standard.optimal_h,
        'metrics': result_standard.metrics
    }
    
    adaptive_results = {
        'denoised': result_adaptive.reconstructed,
        'optimal_h': result_adaptive.optimal_h,
        'adapted_gamma': scsa_adaptive.scsa1d.gamma,
        'metrics': result_adaptive.metrics
    }
    
    # Plot if requested
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axes[0].plot(x, signal, 'k-', linewidth=2, label='Original')
        axes[0].plot(x, noisy, 'r-', alpha=0.3, label='Noisy')
        axes[0].legend()
        axes[0].set_title("Original and Noisy Signals")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(x, signal, 'k-', linewidth=2, label='Original')
        axes[1].plot(x, result_standard.reconstructed, 'b--', label='Standard SCSA')
        axes[1].legend()
        axes[1].set_title(f"Standard SCSA (h={result_standard.optimal_h:.2f})")
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(x, signal, 'k-', linewidth=2, label='Original')
        axes[2].plot(x, result_adaptive.reconstructed, 'g--', label='Adaptive SCSA')
        axes[2].legend()
        axes[2].set_title(f"Adaptive SCSA (γ={scsa_adaptive.scsa1d.gamma:.2f}, h={result_adaptive.optimal_h:.2f})")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return standard_results, adaptive_results


def robust_denoising(signal: Optional[np.ndarray] = None,
                    outlier_fraction: float = 0.05,
                    plot: bool = True) -> Dict[str, Any]:
    """
    Demonstrate robust SCSA with outlier handling.
    
    Parameters
    ----------
    signal : np.ndarray, optional
        Input signal. If None, generates a test signal
    outlier_fraction : float
        Fraction of points to make outliers
    plot : bool
        Whether to display plots
        
    Returns
    -------
    dict
        Results dictionary
    """
    if signal is None:
        x = np.linspace(-10, 10, 500)
        signal = SignalGenerator.double_well(x, separation=3, depth=50)
    
    # Add noise and outliers
    noisy = add_noise(signal, snr_db=20, seed=42)
    
    # Add outliers
    n_outliers = int(len(noisy) * outlier_fraction)
    outlier_indices = np.random.choice(len(noisy), size=n_outliers, replace=False)
    noisy[outlier_indices] += np.random.choice([-1, 1], size=n_outliers) * np.random.uniform(10, 20, size=n_outliers)
    
    # Apply robust SCSA
    scsa_robust = RobustSCSA(gamma=0.5, outlier_threshold=3.0)
    result = scsa_robust.denoise(np.abs(noisy), handle_outliers=True)
    
    # Plot if requested
    if plot:
        viz = SCSAVisualizer()
        fig = viz.plot_1d_comparison(
            np.abs(signal),
            result.reconstructed,
            np.abs(noisy),
            title=f"Robust SCSA with {outlier_fraction*100:.0f}% Outliers"
        )
        plt.show()
    
    return {
        'original': signal,
        'noisy_with_outliers': noisy,
        'denoised': result.reconstructed,
        'outlier_indices': outlier_indices,
        'metrics': result.metrics
    }


def generate_test_signals(n_samples: int = 500) -> Dict[str, np.ndarray]:
    """
    Generate various test signals for examples.
    
    Parameters
    ----------
    n_samples : int
        Number of samples in each signal
        
    Returns
    -------
    dict
        Dictionary of test signals
    """
    x = np.linspace(-10, 10, n_samples)
    
    signals = {
        'sech_squared': SignalGenerator.sech_squared(x),
        'double_well': SignalGenerator.double_well(x, separation=3, depth=50),
        'gaussian_mixture': SignalGenerator.gaussian_mixture(
            x, centers=[-5, 0, 5], amplitudes=[1, 2, 1.5], widths=[1, 0.5, 0.8]
        ),
        'chirp': SignalGenerator.chirp(x, f0=0.1, f1=2, method='linear'),
        'step_function': SignalGenerator.step_function(
            x, [(-5, 1), (0, 3), (5, 2)]
        )
    }
    
    return signals


def generate_test_image(size: int = 100, pattern: str = 'gaussian') -> np.ndarray:
    """
    Generate test images for examples.
    
    Parameters
    ----------
    size : int
        Image size (will be size x size)
    pattern : str
        Pattern type ('gaussian', 'checkerboard', 'rings', 'gradient')
        
    Returns
    -------
    np.ndarray
        Generated test image
    """
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    if pattern == 'gaussian':
        return np.exp(-(X**2 + Y**2) / 4)
    elif pattern == 'checkerboard':
        return (np.sin(3 * X) * np.sin(3 * Y) > 0).astype(float)
    elif pattern == 'rings':
        R = np.sqrt(X**2 + Y**2)
        return 0.5 * (1 + np.sin(5 * R))
    elif pattern == 'gradient':
        return (X + Y) / 10 + 0.5
    elif pattern == 'multi_gaussian':
        image = np.zeros_like(X)
        centers = [(-2, -2), (2, 2), (-2, 2), (2, -2), (0, 0)]
        for cx, cy in centers:
            image += np.exp(-((X - cx)**2 + (Y - cy)**2) / 1.5)
        return image / image.max()
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def parameter_optimization(signal: Optional[np.ndarray] = None,
                         gamma_range: Tuple[float, float] = (0.1, 2.0),
                         n_gammas: int = 10) -> Dict[str, Any]:
    """
    Demonstrate parameter optimization process.
    
    Parameters
    ----------
    signal : np.ndarray, optional
        Input signal
    gamma_range : tuple
        Range of gamma values to test
    n_gammas : int
        Number of gamma values to test
        
    Returns
    -------
    dict
        Optimization results
    """
    if signal is None:
        x = np.linspace(-10, 10, 500)
        signal = SignalGenerator.sech_squared(x)
    
    # Add noise
    noisy = add_noise(signal, snr_db=20, seed=42)
    
    # Test different gamma values
    gammas = np.linspace(gamma_range[0], gamma_range[1], n_gammas)
    results = []
    
    for gamma in gammas:
        scsa = SCSA1D(gamma=gamma)
        result = scsa.filter_with_optimal_h(np.abs(noisy))
        
        results.append({
            'gamma': gamma,
            'optimal_h': result.optimal_h,
            'mse': result.metrics['mse'],
            'psnr': result.metrics['psnr'],
            'num_eigenvalues': result.num_eigenvalues
        })
    
    # Find best gamma
    best_idx = np.argmax([r['psnr'] for r in results])
    best_result = results[best_idx]
    
    return {
        'all_results': results,
        'best_gamma': best_result['gamma'],
        'best_h': best_result['optimal_h'],
        'best_psnr': best_result['psnr'],
        'gammas_tested': gammas
    }


def quick_demo():
    """
    Quick demonstration of SCSA capabilities.
    
    This function runs a simple demo showing:
    1. 1D signal denoising
    2. 2D image reconstruction
    3. Performance metrics
    """
    print("=" * 60)
    print("SCSA Quick Demo")
    print("=" * 60)
    
    # 1D Demo
    print("\n1. 1D Signal Denoising")
    print("-" * 40)
    results_1d = basic_1d_denoising(snr_db=15, plot=True)
    print(f"  Optimal h: {results_1d['optimal_h']:.3f}")
    print(f"  PSNR: {results_1d['metrics']['psnr']:.2f} dB")
    print(f"  MSE: {results_1d['metrics']['mse']:.6f}")
    
    # 2D Demo
    print("\n2. 2D Image Reconstruction")
    print("-" * 40)
    results_2d = basic_2d_reconstruction(snr_db=12, plot=True)
    print(f"  PSNR: {results_2d['metrics']['psnr']:.2f} dB")
    print(f"  MSE: {results_2d['metrics']['mse']:.6f}")
    if 'ssim' in results_2d['metrics']:
        print(f"  SSIM: {results_2d['metrics']['ssim']:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo complete! Explore more examples in the examples folder.")
    
    return results_1d, results_2d


def run_all_examples(plot: bool = False):
    """
    Run all example functions.
    
    Parameters
    ----------
    plot : bool
        Whether to show plots (default False to avoid too many windows)
        
    Returns
    -------
    dict
        Results from all examples
    """
    print("Running all SCSA examples...")
    print("=" * 60)
    
    all_results = {}
    
    # Basic denoising
    print("\n1. Basic 1D Denoising...")
    all_results['basic_1d'] = basic_1d_denoising(plot=plot)
    print(f"   PSNR: {all_results['basic_1d']['metrics']['psnr']:.2f} dB")
    
    # Basic 2D reconstruction
    print("\n2. Basic 2D Reconstruction...")
    all_results['basic_2d'] = basic_2d_reconstruction(plot=plot)
    print(f"   PSNR: {all_results['basic_2d']['metrics']['psnr']:.2f} dB")
    
    # Adaptive filtering
    print("\n3. Adaptive Filtering...")
    std_results, adapt_results = adaptive_filtering(plot=plot)
    all_results['adaptive'] = {
        'standard': std_results,
        'adaptive': adapt_results
    }
    print(f"   Standard MSE: {std_results['metrics']['mse']:.6f}")
    print(f"   Adaptive MSE: {adapt_results['metrics']['mse']:.6f}")
    
    # Robust denoising
    print("\n4. Robust Denoising...")
    all_results['robust'] = robust_denoising(plot=plot)
    print(f"   PSNR: {all_results['robust']['metrics']['psnr']:.2f} dB")
    
    # Parameter optimization
    print("\n5. Parameter Optimization...")
    all_results['optimization'] = parameter_optimization()
    print(f"   Best gamma: {all_results['optimization']['best_gamma']:.2f}")
    print(f"   Best PSNR: {all_results['optimization']['best_psnr']:.2f} dB")
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    
    return all_results


# Module-level constants for easy access
AVAILABLE_PATTERNS = ['gaussian', 'checkerboard', 'rings', 'gradient', 'multi_gaussian']
AVAILABLE_SIGNALS = ['sech_squared', 'double_well', 'gaussian_mixture', 'chirp', 'step_function']

__all__ = [
    # From signal_denoising.py
    'example_basic_denoising',
    'example_multi_noise_comparison',
    'example_parameter_optimization',
    'example_adaptive_filtering',
    
    # From image_reconstruction.py
    'example_basic_2d_reconstruction',
    'example_windowed_processing',
    'example_different_patterns',
    'example_multiscale_processing',
    
    # From benchmark.py
    'benchmark_1d_methods',
    'benchmark_2d_window_sizes',
    'benchmark_gamma_values',
    'benchmark_noise_levels',
    'benchmark_memory_usage',
    'comprehensive_benchmark',
    
    # Utility functions from __init__.py itself
    'generate_test_signals',
    'generate_test_image',
    'quick_demo',
    'run_all_examples',
    
    # Constants
    'AVAILABLE_PATTERNS',
    'AVAILABLE_SIGNALS'
]