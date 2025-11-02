"""
Batch processing script for multiple signals/images using SCSA.
"""

import os
import numpy as np
import glob
from pathlib import Path
import argparse
import json
from datetime import datetime
from tqdm import tqdm
from pyscsa import SCSA1D, SCSA2D
from pyscsa.metrics import QualityMetrics
import matplotlib.pyplot as plt


class BatchProcessor:
    """Batch processing for SCSA."""
    
    def __init__(self, input_dir, output_dir, config_file=None):
        """
        Initialize batch processor.
        
        Parameters
        ----------
        input_dir : str
            Directory containing input files
        output_dir : str
            Directory for output files
        config_file : str, optional
            Configuration file path
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.default_config()
        
        # Initialize processors
        self.scsa1d = SCSA1D(gamma=self.config['gamma_1d'])
        self.scsa2d = SCSA2D(gamma=self.config['gamma_2d'])
        
        # Results storage
        self.results = []
    
    def default_config(self):
        """Default configuration."""
        return {
            'gamma_1d': 0.5,
            'gamma_2d': 2.0,
            'h_1d': None,  # Auto-optimize
            'h_2d': 10.0,
            'window_size': 8,
            'method_2d': 'windowed',
            'file_patterns': ['*.npy', '*.npz'],
            'save_metrics': True,
            'save_plots': True
        }
    
    def process_1d_signal(self, signal, filename):
        """Process 1D signal."""
        print(f"  Processing 1D signal: {filename}")
        
        if self.config['h_1d'] is None:
            result = self.scsa1d.filter_with_optimal_h(np.abs(signal))
        else:
            result = self.scsa1d.reconstruct(np.abs(signal), h=self.config['h_1d'])
        
        return result
    
    def process_2d_image(self, image, filename):
        """Process 2D image."""
        print(f"  Processing 2D image: {filename}")
        
        if self.config['method_2d'] == 'windowed':
            denoised = self.scsa2d.reconstruct_windowed(
                np.abs(image),
                h=self.config['h_2d'],
                window_size=self.config['window_size']
            )
            # Create result object for consistency
            result = type('Result', (), {
                'reconstructed': denoised,
                'metrics': QualityMetrics.compute_all(image, denoised)
            })()
        else:
            result = self.scsa2d.reconstruct(np.abs(image), h=self.config['h_2d'])
        
        return result
    
    def process_file(self, filepath):
        """Process a single file."""
        filename = filepath.name
        
        try:
            # Load data
            if filepath.suffix == '.npy':
                data = np.load(filepath)
            elif filepath.suffix == '.npz':
                npz_file = np.load(filepath)
                # Take the first array
                data = npz_file[list(npz_file.keys())[0]]
            else:
                print(f"  Unsupported file format: {filename}")
                return None
            
            # Determine dimensionality and process
            if data.ndim == 1:
                result = self.process_1d_signal(data, filename)
                data_type = '1D'
            elif data.ndim == 2:
                result = self.process_2d_image(data, filename)
                data_type = '2D'
            else:
                print(f"  Unsupported dimensionality ({data.ndim}D): {filename}")
                return None
            
            # Save output
            output_path = self.output_dir / f"denoised_{filename}"
            np.save(output_path, result.reconstructed)
            
            # Store results
            result_info = {
                'filename': filename,
                'type': data_type,
                'input_shape': data.shape,
                'timestamp': datetime.now().isoformat()
            }
            
            if hasattr(result, 'metrics'):
                result_info['metrics'] = {
                    k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                    for k, v in result.metrics.items()
                }
            
            if hasattr(result, 'optimal_h'):
                result_info['optimal_h'] = float(result.optimal_h)
            
            self.results.append(result_info)
            
            # Generate plot if requested
            if self.config['save_plots']:
                self.save_plot(data, result.reconstructed, filename, data_type)
            
            print(f"  ✓ Processed successfully")
            return result
            
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {str(e)}")
            return None
    
    def save_plot(self, original, reconstructed, filename, data_type):
        """Save comparison plot."""
        plot_path = self.output_dir / f"plot_{filename.replace('.npy', '.png')}"
        
        if data_type == '1D':
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].plot(original, 'b-', label='Original')
            axes[0].plot(reconstructed, 'r--', label='SCSA')
            axes[0].set_title('Signal Comparison')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(np.abs(original - reconstructed))
            axes[1].set_title('Absolute Error')
            axes[1].grid(True, alpha=0.3)
            
        else:  # 2D
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            im1 = axes[0].imshow(original, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046)
            
            im2 = axes[1].imshow(reconstructed, cmap='gray')
            axes[1].set_title('SCSA Denoised')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046)
            
            im3 = axes[2].imshow(np.abs(original - reconstructed), cmap='hot')
            axes[2].set_title('Absolute Error')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        plt.suptitle(f'SCSA Processing: {filename}')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def run(self):
        """Run batch processing."""
        print("=" * 60)
        print("SCSA Batch Processing")
        print("=" * 60)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Configuration: {self.config}")
        print("-" * 60)
        
        # Find all files
        files = []
        for pattern in self.config['file_patterns']:
            files.extend(list(self.input_dir.glob(pattern)))
        
        if not files:
            print("No files found to process.")
            return
        
        print(f"Found {len(files)} files to process\n")
        
        # Process files
        for filepath in tqdm(files, desc="Processing files"):
            print(f"\nProcessing: {filepath.name}")
            self.process_file(filepath)
        
        # Save results summary
        if self.config['save_metrics'] and self.results:
            summary_path = self.output_dir / 'processing_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\nResults summary saved to: {summary_path}")
        
        # Print summary statistics
        self.print_summary()
    
    def print_summary(self):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("Processing Summary")
        print("=" * 60)
        
        if not self.results:
            print("No results to summarize.")
            return
        
        # Count by type
        type_counts = {}
        for result in self.results:
            data_type = result['type']
            type_counts[data_type] = type_counts.get(data_type, 0) + 1
        
        print(f"Total files processed: {len(self.results)}")
        for data_type, count in type_counts.items():
            print(f"  {data_type} signals: {count}")
        
        # Average metrics
        if any('metrics' in r for r in self.results):
            print("\nAverage metrics:")
            
            for data_type in type_counts.keys():
                type_results = [r for r in self.results if r['type'] == data_type and 'metrics' in r]
                if type_results:
                    print(f"\n  {data_type} signals:")
                    
                    # Collect all metric names
                    metric_names = set()
                    for r in type_results:
                        metric_names.update(r['metrics'].keys())
                    
                    for metric in metric_names:
                        values = [r['metrics'][metric] for r in type_results 
                                 if metric in r['metrics'] and isinstance(r['metrics'][metric], (int, float))]
                        if values:
                            avg_value = np.mean(values)
                            print(f"    {metric}: {avg_value:.4f}")


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(description='SCSA Batch Processing')
    parser.add_argument('input_dir', help='Input directory path')
    parser.add_argument('output_dir', help='Output directory path')
    parser.add_argument('--config', help='Configuration file (JSON)', default=None)
    parser.add_argument('--gamma-1d', type=float, default=0.5, help='Gamma for 1D signals')
    parser.add_argument('--gamma-2d', type=float, default=2.0, help='Gamma for 2D images')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--no-metrics', action='store_true', help='Disable metrics saving')
    
    args = parser.parse_args()
    
    # Create processor
    processor = BatchProcessor(args.input_dir, args.output_dir, args.config)
    
    # Override config with command-line arguments
    if args.gamma_1d:
        processor.config['gamma_1d'] = args.gamma_1d
    if args.gamma_2d:
        processor.config['gamma_2d'] = args.gamma_2d
    if args.no_plots:
        processor.config['save_plots'] = False
    if args.no_metrics:
        processor.config['save_metrics'] = False
    
    # Run processing
    processor.run()


if __name__ == "__main__":
    # Example usage without command-line arguments
    import tempfile
    
    # Create temporary directories for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / 'input'
        output_dir = Path(temp_dir) / 'output'
        input_dir.mkdir()
        
        # Generate sample data
        print("Generating sample data...")
        
        # 1D signals
        for i in range(5):
            x = np.linspace(-10, 10, 500)
            signal = -2 * (1/np.cosh(x))**2 + 0.1 * np.random.randn(500)
            np.save(input_dir / f'signal_{i}.npy', signal)
        
        # 2D images
        for i in range(3):
            image = np.random.randn(50, 50)
            np.save(input_dir / f'image_{i}.npy', image)
        
        # Run batch processing
        processor = BatchProcessor(input_dir, output_dir)
        processor.run()
        
        print(f"\nOutput files saved to: {output_dir}")
        print("Demo complete!")