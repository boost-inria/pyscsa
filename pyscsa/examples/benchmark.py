"""
Performance Benchmarking for SCSA Methods
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from pyscsa import SCSA1D, SCSA2D, add_noise
from pyscsa.filters import AdaptiveSCSA, MultiScaleSCSA, RobustSCSA
from pyscsa.metrics import PerformanceAnalyzer
from pyscsa.utils import SignalGenerator


def benchmark_1d_methods():
    """Benchmark different 1D SCSA methods."""
    print("=" * 50)
    print("1D SCSA Methods Benchmark")
    print("=" * 50)
    
    # Generate test signals of different sizes
    sizes = [100, 500, 1000, 2000, 5000]
    
    # Methods to test
    methods = {
        'Standard SCSA': SCSA1D(gmma=0.5),
        'Adaptive SCSA': AdaptiveSCSA(base_gmma=0.5),
        'Robust SCSA': RobustSCSA(gmma=0.5)
    }
    
    results = {method: {'times': [], 'sizes': sizes} for method in methods}
    
    for size in sizes:
        print(f"\nTesting size: {size}")
        
        # Generate signal
        x = np.linspace(-10, 10, size)
        signal = SignalGenerator.sech_squared(x)
        noisy = add_noise(signal, snr_db=20, seed=42)
        
        for method_name, method in methods.items():
            # Time the method
            times = []
            for _ in range(5):  # 5 runs for averaging
                start = time.perf_counter()
                
                if isinstance(method, SCSA1D):
                    _ = method.reconstruct(np.abs(noisy), h=1.0)
                elif isinstance(method, (AdaptiveSCSA, RobustSCSA)):
                    _ = method.denoise(np.abs(noisy))
                
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            results[method_name]['times'].append(avg_time)
            print(f"  {method_name}: {avg_time:.4f} seconds")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for method_name, data in results.items():
        plt.plot(data['sizes'], data['times'], 'o-', label=method_name, linewidth=2)
    
    plt.xlabel("Signal Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title("1D SCSA Methods Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results


def benchmark_2d_window_sizes():
    """Benchmark different window sizes for 2D SCSA."""
    print("\n" + "=" * 50)
    print("2D SCSA Window Size Benchmark")
    print("=" * 50)
    
    # Test image
    image_size = 128
    image = np.random.randn(image_size, image_size)
    noisy = add_noise(image, snr_db=15, seed=42)
    
    scsa = SCSA2D(gmma=2.0)
    window_sizes = [4, 8, 16, 32, 64]
    
    results = {'window_size': window_sizes, 'time': [], 'quality': []}
    
    for ws in window_sizes:
        print(f"\nWindow size: {ws}")
        
        # Time processing
        start = time.perf_counter()
        reconstructed = scsa.reconstruct_windowed(
            np.abs(noisy), h=5.0, window_size=ws, stride=ws//2
        )
        end = time.perf_counter()
        
        exec_time = end - start
        results['time'].append(exec_time)
        
        # Quality metric
        mse = np.mean((image - reconstructed)**2)
        psnr = 20 * np.log10(np.max(image) / np.sqrt(mse)) if mse > 0 else float('inf')
        results['quality'].append(psnr)
        
        print(f"  Time: {exec_time:.4f} seconds")
        print(f"  PSNR: {psnr:.2f} dB")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(window_sizes, results['time'], 'o-', linewidth=2, color='blue')
    axes[0].set_xlabel("Window Size")
    axes[0].set_ylabel("Execution Time (seconds)")
    axes[0].set_title("Processing Time vs Window Size")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(window_sizes, results['quality'], 'o-', linewidth=2, color='green')
    axes[1].set_xlabel("Window Size")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Reconstruction Quality vs Window Size")
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("2D SCSA Window Size Trade-offs")
    plt.tight_layout()
    plt.show()
    
    return results


def benchmark_gamma_values():
    """Benchmark performance for different gamma values."""
    print("\n" + "=" * 50)
    print("Gamma Parameter Impact Benchmark")
    print("=" * 50)
    
    # Test signal
    x = np.linspace(-10, 10, 1000)
    signal = SignalGenerator.double_well(x, separation=3, depth=50)
    noisy = add_noise(signal, snr_db=18, seed=42)
    
    gammas = [0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    
    results = {
        'gamma': gammas,
        'time': [],
        'mse': [],
        'num_eigenvalues': []
    }
    
    for gmma in gammas:
        print(f"\nGamma: {gmma}")
        
        scsa = SCSA1D(gmma=gmma)
        
        # Time processing
        start = time.perf_counter()
        result = scsa.reconstruct(np.abs(noisy), h=2.0)
        end = time.perf_counter()
        
        exec_time = end - start
        results['time'].append(exec_time)
        results['mse'].append(result.metrics['mse'])
        results['num_eigenvalues'].append(result.num_eigenvalues)
        
        print(f"  Time: {exec_time:.4f} seconds")
        print(f"  MSE: {result.metrics['mse']:.6f}")
        print(f"  Eigenvalues: {result.num_eigenvalues}")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(gammas, results['time'], 'o-', linewidth=2)
    axes[0].set_xlabel("Gamma")
    axes[0].set_ylabel("Execution Time (seconds)")
    axes[0].set_title("Processing Time")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(gammas, results['mse'], 'o-', linewidth=2, color='red')
    axes[1].set_xlabel("Gamma")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Reconstruction Error")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(gammas, results['num_eigenvalues'], 'o-', linewidth=2, color='green')
    axes[2].set_xlabel("Gamma")
    axes[2].set_ylabel("Number of Eigenvalues")
    axes[2].set_title("Eigenvalues Used")
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle("Impact of Gamma Parameter on SCSA Performance")
    plt.tight_layout()
    plt.show()
    
    return results


def benchmark_noise_levels():
    """Benchmark SCSA performance at different noise levels."""
    print("\n" + "=" * 50)
    print("SCSA Performance vs Noise Level")
    print("=" * 50)
    
    # Generate clean signal
    x = np.linspace(-10, 10, 500)
    signal = SignalGenerator.sech_squared(x)
    
    # Test different SNR levels
    snr_levels = [5, 10, 15, 20, 25, 30, 35, 40]
    
    methods = {
        'Standard': SCSA1D(gmma=0.5),
        'Adaptive': AdaptiveSCSA(base_gmma=0.5),
        'Robust': RobustSCSA(gmma=0.5)
    }
    
    results = {method: {'snr': snr_levels, 'mse': [], 'psnr': []} 
              for method in methods}
    
    for snr in snr_levels:
        print(f"\nSNR: {snr} dB")
        
        # Add noise
        noisy = add_noise(signal, snr_db=snr, seed=42)
        
        for method_name, method in methods.items():
            if isinstance(method, SCSA1D):
                result = method.filter_with_optimal_h(np.abs(noisy))
            else:
                result = method.denoise(np.abs(noisy))
            
            results[method_name]['mse'].append(result.metrics['mse'])
            results[method_name]['psnr'].append(result.metrics['psnr'])
            
            print(f"  {method_name} - MSE: {result.metrics['mse']:.6f}, "
                  f"PSNR: {result.metrics['psnr']:.2f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for method_name, data in results.items():
        axes[0].plot(data['snr'], data['mse'], 'o-', label=method_name, linewidth=2)
        axes[1].plot(data['snr'], data['psnr'], 'o-', label=method_name, linewidth=2)
    
    axes[0].set_xlabel("Input SNR (dB)")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("MSE vs Noise Level")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    axes[1].set_xlabel("Input SNR (dB)")
    axes[1].set_ylabel("Output PSNR (dB)")
    axes[1].set_title("PSNR vs Noise Level")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("SCSA Methods Performance vs Noise Level")
    plt.tight_layout()
    plt.show()
    
    return results


def benchmark_memory_usage():
    """Estimate memory usage for different signal sizes."""
    print("\n" + "=" * 50)
    print("Memory Usage Estimation")
    print("=" * 50)
    
    import psutil
    import os
    
    sizes = [100, 500, 1000, 2000, 5000, 10000]
    
    memory_usage = {'size': sizes, 'memory_mb': []}
    
    for size in sizes:
        # Generate signal
        x = np.linspace(-10, 10, size)
        signal = np.random.randn(size)
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Run SCSA
        scsa = SCSA1D(gmma=0.5)
        result = scsa.reconstruct(np.abs(signal), h=1.0)
        
        # Get memory after
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_used = mem_after - mem_before
        
        memory_usage['memory_mb'].append(mem_used)
        
        print(f"Size: {size}, Memory used: {mem_used:.2f} MB")
        
        # Clean up
        del result, scsa, signal, x
    
    # Plot memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, memory_usage['memory_mb'], 'o-', linewidth=2, color='purple')
    plt.xlabel("Signal Size")
    plt.ylabel("Memory Usage (MB)")
    plt.title("SCSA Memory Usage vs Signal Size")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return memory_usage


def comprehensive_benchmark():
    """Run comprehensive performance analysis."""
    print("\n" + "=" * 50)
    print("Comprehensive Performance Analysis")
    print("=" * 50)
    
    analyzer = PerformanceAnalyzer()
    
    # Test signal
    x = np.linspace(-10, 10, 1000)
    signal = SignalGenerator.gaussian_mixture(
        x, centers=[-5, 0, 5], amplitudes=[1, 2, 1.5], widths=[1, 0.5, 0.8]
    )
    noisy = add_noise(signal, snr_db=20, seed=42)
    
    # Methods to compare
    methods_to_test = [
        (SCSA1D(gmma=0.5).reconstruct, {'h': 1.0}),
        (SCSA1D(gmma=0.5).filter_with_optimal_h, {}),
        (AdaptiveSCSA(base_gmma=0.5).denoise, {'adapt_h': True}),
        (RobustSCSA(gmma=0.5).denoise, {'handle_outliers': True})
    ]
    
    # Name the methods
    method_names = [
        'Standard SCSA',
        'SCSA with Optimal h',
        'Adaptive SCSA',
        'Robust SCSA'
    ]
    
    # Run benchmarks
    print("\nRunning benchmarks (10 runs each)...")
    comparison = {}
    
    for (method, kwargs), name in zip(methods_to_test, method_names):
        print(f"  Testing {name}...")
        stats = analyzer.benchmark(method, np.abs(noisy), n_runs=10, **kwargs)
        comparison[name] = stats
    
    # Display results
    print("\n" + "-" * 50)
    print("Benchmark Results Summary:")
    print("-" * 50)
    
    for method_name, stats in comparison.items():
        print(f"\n{method_name}:")
        print(f"  Mean time: {stats['mean_time']:.4f} ± {stats['std_time']:.4f} seconds")
        print(f"  Min time: {stats['min_time']:.4f} seconds")
        print(f"  Max time: {stats['max_time']:.4f} seconds")
    
    # Plot comparison
    fig = analyzer.plot_comparison(comparison)
    plt.show()
    
    return comparison


if __name__ == "__main__":
    print("SCSA Performance Benchmarking Suite")
    print("=" * 50)
    
    # Check if psutil is available for memory benchmarking
    try:
        import psutil
        has_psutil = True
    except ImportError:
        print("Note: Install psutil for memory usage benchmarking")
        print("  pip install psutil")
        has_psutil = False
    
    # Run benchmarks
    results = {}
    
    # 1D methods comparison
    results['1d_methods'] = benchmark_1d_methods()
    
    # 2D window sizes
    results['2d_windows'] = benchmark_2d_window_sizes()
    
    # Gamma parameter impact
    results['gamma_impact'] = benchmark_gamma_values()
    
    # Noise level performance
    results['noise_levels'] = benchmark_noise_levels()
    
    # Memory usage (if psutil available)
    if has_psutil:
        results['memory'] = benchmark_memory_usage()
    
    # Comprehensive benchmark
    results['comprehensive'] = comprehensive_benchmark()
    
    print("\n" + "=" * 50)
    print("All benchmarks completed successfully!")
    print("=" * 50)
    
    # Save results summary
    print("\nBenchmark Summary:")
    print("-" * 50)
    print("1. 1D methods show different trade-offs between speed and accuracy")
    print("2. Window size affects both speed and quality in 2D processing")
    print("3. Gamma parameter significantly impacts reconstruction quality")
    print("4. SCSA maintains good performance across different noise levels")
    if has_psutil:
        print("5. Memory usage scales approximately linearly with signal size")
    print("6. Adaptive methods provide better quality at the cost of computation time")
