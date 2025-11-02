"""
Image Reconstruction Examples using SCSA
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscsa import SCSA2D, add_noise
from pyscsa.visualization import SCSAVisualizer
from pyscsa.metrics import QualityMetrics
from pyscsa.filters import MultiScaleSCSA


def generate_test_image(size=100, pattern='gaussian'):
    """Generate various test images."""
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
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def example_basic_2d_reconstruction():
    """Basic 2D image reconstruction."""
    print("=" * 50)
    print("Basic 2D Image Reconstruction")
    print("=" * 50)
    
    # Generate test image
    image = generate_test_image(100, 'gaussian')
    
    # Add noise
    noisy = add_noise(image, snr_db=15, seed=42)
    
    # Reconstruct
    scsa = SCSA2D(gmma=2.0)
    result = scsa.reconstruct(noisy, h=2.0)
    
    # Metrics
    metrics = QualityMetrics.compute_all(image, result.reconstructed)
    
    print(f"\nReconstruction Results:")
    print(f"  h parameter: 5.0")
    print(f"  Number of eigenvalues: {result.num_eigenvalues}")
    print(f"  MSE: {metrics['mse'].value:.6f}")
    print(f"  PSNR: {metrics['psnr'].value:.2f} dB")
    print(f"  SSIM: {metrics['ssim'].value:.4f}")
    
    # Visualize
    viz = SCSAVisualizer()
    fig = viz.plot_2d_comparison(
        image,
        result.reconstructed,
        noisy,
        title="SCSA 2D Reconstruction - Gaussian Pattern",
        metrics={k: v.value for k, v in metrics.items()}
    )
    plt.show()
    
    return result


def example_windowed_processing():
    """Compare standard vs windowed SCSA processing."""
    print("\n" + "=" * 50)
    print("Windowed vs Standard SCSA Processing")
    print("=" * 50)
    
    # Generate larger image
    image = generate_test_image(200, 'rings')
    noisy = add_noise(image, snr_db=12, seed=42)
    
    scsa = SCSA2D(gmma=2.0)
    
    # Standard processing
    print("\nStandard SCSA processing...")
    result_standard = scsa.reconstruct(noisy, h=2.0)
    
    # Windowed processing with different window sizes
    window_sizes = [8, 16, 32]
    results_windowed = {}
    
    for ws in window_sizes:
        print(f"Windowed SCSA (window={ws})...")
        reconstructed = scsa.reconstruct_windowed(
            noisy, h=2.0, window_size=ws, stride=ws//2
        )
        results_windowed[ws] = reconstructed
    
    # Compare metrics
    metrics_standard = QualityMetrics.compute_all(image, result_standard.reconstructed)
    print(f"\nStandard - PSNR: {metrics_standard['psnr'].value:.2f} dB, "
          f"SSIM: {metrics_standard['ssim'].value:.4f}")
    
    for ws, recon in results_windowed.items():
        metrics = QualityMetrics.compute_all(image, recon)
        print(f"Window {ws} - PSNR: {metrics['psnr'].value:.2f} dB, "
              f"SSIM: {metrics['ssim'].value:.4f}")
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title("Noisy")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(result_standard.reconstructed, cmap='gray')
    axes[0, 2].set_title("Standard SCSA")
    axes[0, 2].axis('off')
    
    for idx, (ws, recon) in enumerate(results_windowed.items()):
        axes[1, idx].imshow(recon, cmap='gray')
        axes[1, idx].set_title(f"Windowed (size={ws})")
        axes[1, idx].axis('off')
    
    plt.suptitle("Standard vs Windowed SCSA Comparison")
    plt.tight_layout()
    plt.show()
    
    return result_standard, results_windowed


def example_different_patterns():
    """Test SCSA on different image patterns."""
    print("\n" + "=" * 50)
    print("SCSA on Different Image Patterns")
    print("=" * 50)
    
    patterns = ['gaussian', 'checkerboard', 'rings', 'gradient']
    scsa = SCSA2D(gmma=2.0)
    viz = SCSAVisualizer()
    
    fig, axes = plt.subplots(len(patterns), 4, figsize=(16, 4*len(patterns)))
    
    for i, pattern in enumerate(patterns):
        # Generate image
        image = generate_test_image(100, pattern)
        noisy = add_noise(image, snr_db=15, seed=42)
        
        # Reconstruct
        denoised = scsa.denoise(noisy, method='windowed', 
                               window_size=10, h=5.0)
        
        # Metrics
        metrics = QualityMetrics.compute_all(image, denoised)
        
        print(f"\n{pattern.capitalize()} pattern:")
        print(f"  PSNR: {metrics['psnr'].value:.2f} dB")
        print(f"  SSIM: {metrics['ssim'].value:.4f}")
        
        # Plot
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title(f"{pattern.capitalize()} - Original")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(noisy, cmap='gray')
        axes[i, 1].set_title("Noisy")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(denoised, cmap='gray')
        axes[i, 2].set_title("SCSA Denoised")
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(np.abs(image - denoised), cmap='hot')
        axes[i, 3].set_title("Error")
        axes[i, 3].axis('off')
    
    plt.suptitle("SCSA Performance on Different Patterns")
    plt.tight_layout()
    plt.show()


def example_multiscale_processing():
    """Demonstrate multi-scale SCSA processing."""
    print("\n" + "=" * 50)
    print("Multi-scale SCSA Processing")
    print("=" * 50)
    
    # Generate complex image
    size = 128
    x = np.linspace(0, 10, size)
    y = np.linspace(0, 10, size)
    X, Y = np.meshgrid(x, y)
    
    # Multi-frequency pattern
    image = (0.5 * np.sin(X) * np.cos(Y) +  # Low frequency
             0.3 * np.sin(5*X) * np.cos(5*Y) +  # Medium frequency
             0.2 * np.sin(10*X) * np.cos(10*Y))  # High frequency
    image = (image - image.min()) / (image.max() - image.min())
    
    # Add noise
    noisy = add_noise(image, snr_db=10, seed=42)
    
    # Standard SCSA
    scsa = SCSA2D(gmma=2.0)
    result_standard = scsa.reconstruct(noisy, h=8.0)
    
    # Multi-scale SCSA
    ms_scsa = MultiScaleSCSA(scales=[1, 2, 4], gmma=2.0)
    result_multiscale = ms_scsa.reconstruct(noisy)
    
    # Metrics
    metrics_std = QualityMetrics.compute_all(image, result_standard.reconstructed)
    metrics_ms = QualityMetrics.compute_all(image, result_multiscale)
    
    print("\nStandard SCSA:")
    print(f"  PSNR: {metrics_std['psnr'].value:.2f} dB")
    print(f"  SSIM: {metrics_std['ssim'].value:.4f}")
    
    print("\nMulti-scale SCSA:")
    print(f"  PSNR: {metrics_ms['psnr'].value:.2f} dB")
    print(f"  SSIM: {metrics_ms['ssim'].value:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title("Noisy")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(result_standard.reconstructed, cmap='gray')
    axes[0, 2].set_title("Standard SCSA")
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(result_multiscale, cmap='gray')
    axes[1, 0].set_title("Multi-scale SCSA")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.abs(image - result_standard.reconstructed), cmap='hot')
    axes[1, 1].set_title("Standard Error")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.abs(image - result_multiscale), cmap='hot')
    axes[1, 2].set_title("Multi-scale Error")
    axes[1, 2].axis('off')
    
    plt.suptitle("Standard vs Multi-scale SCSA")
    plt.tight_layout()
    plt.show()
    
    return result_standard, result_multiscale


if __name__ == "__main__":
    print("SCSA Image Reconstruction Examples")
    print("=" * 50)
    
    # Basic reconstruction
    result1 = example_basic_2d_reconstruction()
    
    # Windowed processing
    result2_std, result2_win = example_windowed_processing()
    
    # Different patterns
    example_different_patterns()
    
    # Multi-scale processing
    #result4_std, result4_ms = example_multiscale_processing()
    
    print("\n" + "=" * 50)
    print("All image examples completed successfully!")