"""
Command-line interface for SCSA library.
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SCSA - Semi-Classical Signal Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Denoise a 1D signal
  scsa-demo denoise1d input.npy output.npy --gamma 0.5 --auto-h
  
  # Reconstruct a 2D image
  scsa-demo denoise2d image.npy output.npy --method windowed --window-size 8
  
  # Run demo
  scsa-demo demo --type 1d
  
  # Benchmark performance
  scsa-demo benchmark --size 1000 --runs 10
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Denoise 1D command
    parser_1d = subparsers.add_parser('denoise1d', help='Denoise 1D signal')
    parser_1d.add_argument('input', type=str, help='Input signal file (.npy)')
    parser_1d.add_argument('output', type=str, help='Output file (.npy)')
    parser_1d.add_argument('--gamma', type=float, default=0.5, help='Gamma parameter')
    parser_1d.add_argument('--h', type=float, default=None, help='h parameter (if not auto)')
    parser_1d.add_argument('--auto-h', action='store_true', help='Automatically optimize h')
    parser_1d.add_argument('--plot', action='store_true', help='Show comparison plot')
    
    # Denoise 2D command
    parser_2d = subparsers.add_parser('denoise2d', help='Denoise 2D image')
    parser_2d.add_argument('input', type=str, help='Input image file (.npy)')
    parser_2d.add_argument('output', type=str, help='Output file (.npy)')
    parser_2d.add_argument('--gamma', type=float, default=2.0, help='Gamma parameter')
    parser_2d.add_argument('--h', type=float, default=10.0, help='h parameter')
    parser_2d.add_argument('--method', choices=['standard', 'windowed'], 
                          default='windowed', help='Processing method')
    parser_2d.add_argument('--window-size', type=int, default=8, help='Window size')
    parser_2d.add_argument('--plot', action='store_true', help='Show comparison plot')
    
    # Demo command
    parser_demo = subparsers.add_parser('demo', help='Run demonstration')
    parser_demo.add_argument('--type', choices=['1d', '2d', 'both'], 
                           default='both', help='Demo type')
    
    # Benchmark command
    parser_bench = subparsers.add_parser('benchmark', help='Run performance benchmark')
    parser_bench.add_argument('--size', type=int, default=1000, 
                            help='Signal/image size')
    parser_bench.add_argument('--runs', type=int, default=10, 
                            help='Number of benchmark runs')
    parser_bench.add_argument('--dim', choices=['1d', '2d'], default='1d',
                            help='Dimensionality')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Import here to avoid circular imports
    from pyscsa import SCSA1D, SCSA2D, add_noise
    from pyscsa.utils import SignalGenerator
    from pyscsa.metrics import PerformanceAnalyzer
    
    if args.command == 'denoise1d':
        # Load signal
        signal = np.load(args.input)
        
        # Create SCSA instance
        scsa = SCSA1D(gmma=args.gamma)
        
        # Process
        if args.auto_h:
            result = scsa.filter_with_c_scsa(signal)
            print(f"Optimal h: {result.optimal_h:.3f}")
        else:
            h = args.h if args.h else 1.0
            result = scsa.reconstruct(signal, h=h)
        
        # Save result
        np.save(args.output, result.reconstructed)
        
        # Print metrics
        print(f"Reconstruction metrics:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Plot if requested
        if args.plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].plot(signal, label='Input')
            axes[0].plot(result.reconstructed, label='SCSA')
            axes[0].legend()
            axes[0].set_title('Signal Comparison')
            
            axes[1].plot(np.abs(signal - result.reconstructed))
            axes[1].set_title('Absolute Error')
            
            plt.tight_layout()
            plt.show()
    
    elif args.command == 'denoise2d':
        # Load image
        image = np.load(args.input)
        
        # Create SCSA instance
        scsa = SCSA2D(gmma=args.gamma)
        
        # Process
        if args.method == 'windowed':
            denoised = scsa.reconstruct_windowed(
                image, h=args.h, window_size=args.window_size
            )
        else:
            result = scsa.reconstruct(image, h=args.h)
            denoised = result.reconstructed
        
        # Save result
        np.save(args.output, denoised)
        
        # Plot if requested
        if args.plot:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            im1 = axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Input')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046)
            
            im2 = axes[1].imshow(denoised, cmap='gray')
            axes[1].set_title('SCSA Denoised')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046)
            
            diff = np.abs(image - denoised)
            im3 = axes[2].imshow(diff, cmap='hot')
            axes[2].set_title('Absolute Difference')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2], fraction=0.046)
            
            plt.tight_layout()
            plt.show()
    
    elif args.command == 'demo':
        if args.type in ['1d', 'both']:
            print("Running 1D demonstration...")
            
            # Generate test signal
            x = np.linspace(-10, 10, 500)
            signal = SignalGenerator.sech_squared(x)
            noisy_signal = add_noise(signal, snr_db=20, seed=42)
            
            # Apply SCSA
            scsa = SCSA1D(gmma=0.5)
            result = scsa.filter_with_c_scsa(noisy_signal)
            
            print(f"1D Results:")
            print(f"  Optimal h: {result.optimal_h:.3f}")
            print(f"  MSE: {result.metrics['mse']:.6f}")
            print(f"  PSNR: {result.metrics['psnr']:.2f} dB")
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].plot(x, signal, 'k-', label='Original', linewidth=2)
            axes[0].plot(x, noisy_signal, 'r-', alpha=0.5, label='Noisy')
            axes[0].plot(x, -result.reconstructed, 'b--', label='SCSA', linewidth=1.5)
            axes[0].legend()
            axes[0].set_title('1D Signal Denoising')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(x, np.abs(noisy_signal - result.reconstructed))
            axes[1].set_title('Reconstruction Error')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        if args.type in ['2d', 'both']:
            print("\nRunning 2D demonstration...")
            
            # Generate test image
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x, y)
            image = np.exp(-(X**2 + Y**2) / 4)
            noisy_image = add_noise(image, snr_db=15, seed=42)
            
            # Apply SCSA
            scsa = SCSA2D(gmma=2.0)
            denoised = scsa.denoise(
                np.abs(noisy_image), 
                method='windowed',
                window_size=10,
                h=5.0
            )
            
            # Calculate metrics
            mse = np.mean((image - denoised)**2)
            psnr = 20 * np.log10(np.max(image) / np.sqrt(mse))
            
            print(f"2D Results:")
            print(f"  MSE: {mse:.6f}")
            print(f"  PSNR: {psnr:.2f} dB")
            
            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            axes[1].imshow(noisy_image, cmap='gray')
            axes[1].set_title('Noisy')
            axes[1].axis('off')
            
            axes[2].imshow(denoised, cmap='gray')
            axes[2].set_title('SCSA Denoised')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
    
    elif args.command == 'benchmark':
        print(f"Running benchmark ({args.runs} runs)...")
        
        analyzer = PerformanceAnalyzer()
        
        if args.dim == '1d':
            # Generate test signal
            signal = np.random.randn(args.size)
            
            # Create SCSA instance
            scsa = SCSA1D(gmma=0.5)
            
            # Benchmark
            stats = analyzer.benchmark(
                scsa.reconstruct,
                signal,
                n_runs=args.runs,
                h=1.0
            )
            
            print(f"\n1D Benchmark Results:")
            print(f"  Signal size: {args.size}")
            print(f"  Mean time: {stats['mean_time']:.4f} seconds")
            print(f"  Std time: {stats['std_time']:.4f} seconds")
            print(f"  Min time: {stats['min_time']:.4f} seconds")
            print(f"  Max time: {stats['max_time']:.4f} seconds")
            
        else:  # 2d
            # Generate test image
            size = int(np.sqrt(args.size))
            image = np.random.randn(size, size)
            
            # Create SCSA instance
            scsa = SCSA2D(gmma=2.0)
            
            # Benchmark
            stats = analyzer.benchmark(
                scsa.reconstruct,
                image,
                n_runs=args.runs,
                h=5.0
            )
            
            print(f"\n2D Benchmark Results:")
            print(f"  Image size: {size}x{size}")
            print(f"  Mean time: {stats['mean_time']:.4f} seconds")
            print(f"  Std time: {stats['std_time']:.4f} seconds")
            print(f"  Min time: {stats['min_time']:.4f} seconds")
            print(f"  Max time: {stats['max_time']:.4f} seconds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
