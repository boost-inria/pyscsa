"""
Signal Denoising Examples using SCSA
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscsa import SCSA1D, add_noise, normalize_signal
from pyscsa.utils import SignalGenerator, NoiseGenerator
from pyscsa.visualization import SCSAVisualizer
from pyscsa.metrics import QualityMetrics


def example_basic_denoising():
    """Basic signal denoising example."""
    print("=" * 50)
    print("Basic Signal Denoising with SCSA")
    print("=" * 50)
    
    # Generate signal
    x = np.linspace(-10, 10, 500)
    signal = SignalGenerator.sech_squared(x, center=0, width=1)
    
    # Ensure signal is positive for SCSA
    min_signal = None
    if signal.min() < 0:
        min_signal = signal.min()
        signal = signal - min_signal
    signal = signal.flatten()
    # Add noise
    noisy = add_noise(signal, snr_db=20, seed=42)
    
    # Denoise
    scsa = SCSA1D(gmma=0.5)
    result = scsa.filter_with_optimal_h(noisy, curvature_weight = 2.0,h_range = [0.2,5])
    
    # Compute metrics
    metrics = QualityMetrics.compute_all(signal, result.reconstructed)
    
    print(f"\nOptimal h: {result.optimal_h:.3f}")
    print(f"Number of eigenvalues: {result.num_eigenvalues}")
    print("\nReconstruction Metrics:")
    for name, metric in metrics.items():
        print(f"  {name.upper()}: {metric.value:.4f} {metric.unit}")
    
    # Visualize
    viz = SCSAVisualizer()
    fig = viz.plot_1d_comparison(
        np.abs(signal), 
        result.reconstructed, 
        noisy,
        title="SCSA Denoising - Sech-squared Signal",
        x_axis=x,
        metrics=result.metrics
    )
    plt.show()
    
    return result


def example_multi_noise_comparison():
    """Compare SCSA performance with different noise types."""
    print("\n" + "=" * 50)
    print("SCSA Performance with Different Noise Types")
    print("=" * 50)
    
    # Generate clean signal
    x = np.linspace(0, 10, 500)
    signal = SignalGenerator.gaussian_mixture(
        x, 
        centers=[2, 5, 8],
        amplitudes=[1, 2, 1.5],
        widths=[0.5, 0.3, 0.4]
    )
    # Ensure signal is positive for SCSA
    min_signal = None
    if signal.min() < 0:
        min_signal = signal.min()
        signal = signal - min_signal
    signal = signal.flatten()
    
    # Different noise types
    noise_types = {
        'Gaussian': NoiseGenerator.gaussian(signal.shape, snr_db=15, 
                                           signal_power=np.mean(signal**2), seed=42),
        'Poisson': NoiseGenerator.poisson(signal.shape, lam=0.5, seed=42) * 0.1,
        'Uniform': NoiseGenerator.uniform(signal.shape, low=-0.2, high=0.2, seed=42)
    }
    
    # Process each noise type
    results = {}
    scsa = SCSA1D(gmma=2)
    
    for noise_name, noise in noise_types.items():
        noisy = signal + noise
        result = scsa.filter_with_optimal_h(noisy, curvature_weight = 2.0,h_range = [0.2,5])
        metrics = QualityMetrics.compute_all(signal, result.reconstructed)
        results[noise_name] = {
            'result': result,
            'metrics': {m.name: m.value for m in metrics.values()}
        }
        
        print(f"\n{noise_name} Noise:")
        print(f"  Optimal h: {result.optimal_h:.3f}")
        print(f"  PSNR: {metrics['psnr'].value:.2f} dB")
        print(f"  MSE: {metrics['mse'].value:.6f}")
    
    # Compare results
    viz = SCSAVisualizer()
    metrics_comparison = {name: res['metrics'] for name, res in results.items()}
    fig = viz.plot_metrics_comparison(metrics_comparison, 
                                     title="SCSA Performance vs Noise Type")
    plt.show()
    
    return results


def example_parameter_optimization():
    """Demonstrate parameter optimization process."""
    print("\n" + "=" * 50)
    print("SCSA Parameter Optimization")
    print("=" * 50)
    
    # Generate signal with variable complexity
    x = np.linspace(0, 20, 1000)
    signal = (20 + 30 * np.sin(2 * np.pi * x / 30) +
              10 * np.sin(2 * np.pi * x / 10) +
              5 * np.sin(2 * np.pi * x / 5))
    # Ensure signal is positive for SCSA
    min_signal = None
    if signal.min() < 0:
        min_signal = signal.min()
        signal = signal - min_signal
    signal = signal.flatten()
    # Add noise
    noisy = add_noise(signal, snr_db=15, seed=42)
    
    # Test different gamma values
    gammas = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    results = {}
    
    for gmma in gammas:
        scsa = SCSA1D(gmma=gmma)
        result = scsa.filter_with_optimal_h(noisy, curvature_weight = 2.0,h_range = [0.2,5])
        results[gmma] = {
            'h_optimal': result.optimal_h,
            'mse': result.metrics['mse'],
            'psnr': result.metrics['psnr']
        }
        print(f"γ={gmma:.1f}: h_opt={result.optimal_h:.2f}, "
              f"MSE={result.metrics['mse']:.4f}, PSNR={result.metrics['psnr']:.2f}")
    
    # Visualize parameter sweep
    viz = SCSAVisualizer()
    sweep_metrics = {
        'MSE': [results[g]['mse'] for g in gammas],
        'PSNR': [results[g]['psnr'] for g in gammas]
    }
    fig = viz.plot_parameter_sweep(
        np.array(gammas),
        sweep_metrics,
        param_name="gamma",
        optimal_value=0.5
    )
    plt.show()
    
    return results


def example_adaptive_filtering():
    """Demonstrate adaptive SCSA filtering."""
    print("\n" + "=" * 50)
    print("Adaptive SCSA Filtering")
    print("=" * 50)
    
    from pyscsa.filters import AdaptiveSCSA
    
    # Generate signal with varying characteristics
    x = np.linspace(0, 30, 1500)
    
    # Smooth section + sharp transitions + oscillations
    signal = np.zeros_like(x)
    signal[x < 10] = 5 * np.exp(-0.3 * x[x < 10])  # Smooth decay
    signal[(x >= 10) & (x < 15)] = 2  # Constant
    signal[(x >= 15) & (x < 20)] = 2 + 3 * np.sin(4 * np.pi * x[(x >= 15) & (x < 20)])  # Oscillations
    signal[x >= 20] = SignalGenerator.step_function(x[x >= 20] - 20, [(5, 3), (8, 1)])
    # Ensure signal is positive for SCSA
    min_signal = None
    if signal.min() < 0:
        min_signal = signal.min()
        signal = signal - min_signal
    signal = signal.flatten()
    # Add noise
    noisy = add_noise(signal, snr_db=18, seed=42)
    
    # Standard SCSA
    scsa_standard = SCSA1D(gmma=0.5)
    result_standard = scsa_standard.filter_with_optimal_h(noisy, curvature_weight = 2.0,h_range = [0.2,5])
    
    # Adaptive SCSA
    scsa_adaptive = AdaptiveSCSA(base_gmma=0.5)
    result_adaptive = scsa_adaptive.denoise(noisy, adapt_h=True, adapt_gmma=True)
    
    # Compare results
    print("\nStandard SCSA:")
    print(f"  Fixed γ: 0.5")
    print(f"  Optimal h: {result_standard.optimal_h:.3f}")
    print(f"  MSE: {result_standard.metrics['mse']:.6f}")
    
    print("\nAdaptive SCSA:")
    print(f"  Adapted γ: {scsa_adaptive.scsa1d.gmma:.3f}")
    print(f"  Optimal h: {result_adaptive.optimal_h:.3f}")
    print(f"  MSE: {result_adaptive.metrics['mse']:.6f}")
    
    # Visualize comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(x, signal, 'k-', label='Original', linewidth=2)
    axes[0].plot(x, noisy, 'r-', alpha=0.3, label='Noisy')
    axes[0].legend()
    axes[0].set_title("Original and Noisy Signals")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x, signal, 'k-', label='Original', linewidth=2)
    axes[1].plot(x, result_standard.reconstructed, 'b--', label='Standard SCSA')
    axes[1].legend()
    axes[1].set_title("Standard SCSA Reconstruction")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(x, signal, 'k-', label='Original', linewidth=2)
    axes[2].plot(x, result_adaptive.reconstructed, 'g--', label='Adaptive SCSA')
    axes[2].legend()
    axes[2].set_title("Adaptive SCSA Reconstruction")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return result_standard, result_adaptive


if __name__ == "__main__":
    # Run all examples
    print("SCSA Signal Denoising Examples")
    print("=" * 50)
    
    # Basic denoising
    result1 = example_basic_denoising()
    
    # Noise type comparison
    results2 = example_multi_noise_comparison()
    
    # Parameter optimization
    results3 = example_parameter_optimization()
    
    # Adaptive filtering
    result4_std, result4_adapt = example_adaptive_filtering()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")