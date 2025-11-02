"""
Performance metrics and analysis tools.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import time


@dataclass
class MetricResult:
    """Container for metric results."""
    value: float
    name: str
    unit: str = ""
    higher_is_better: bool = True


class QualityMetrics:
    """Comprehensive quality metrics for signal/image assessment."""
    
    @staticmethod
    def mse(original: np.ndarray, reconstructed: np.ndarray) -> MetricResult:
        """Mean Squared Error."""
        value = np.mean((original - reconstructed)**2)
        return MetricResult(value, "MSE", "", False)
    
    @staticmethod
    def rmse(original: np.ndarray, reconstructed: np.ndarray) -> MetricResult:
        """Root Mean Squared Error."""
        value = np.sqrt(np.mean((original - reconstructed)**2))
        return MetricResult(value, "RMSE", "", False)
    
    @staticmethod
    def psnr(original: np.ndarray, reconstructed: np.ndarray) -> MetricResult:
        """Peak Signal-to-Noise Ratio."""
        mse = np.mean((original - reconstructed)**2)
        if mse == 0:
            value = float('inf')
        else:
            max_val = np.max(original)
            value = 20 * np.log10(max_val / np.sqrt(mse))
        return MetricResult(value, "PSNR", "dB", True)
    
    @staticmethod
    def snr(original: np.ndarray, reconstructed: np.ndarray) -> MetricResult:
        """Signal-to-Noise Ratio."""
        signal_power = np.mean(original**2)
        noise_power = np.mean((original - reconstructed)**2)
        if noise_power == 0:
            value = float('inf')
        else:
            value = 10 * np.log10(signal_power / noise_power)
        return MetricResult(value, "SNR", "dB", True)
    
    @staticmethod
    def ssim(original: np.ndarray, reconstructed: np.ndarray) -> MetricResult:
        """Structural Similarity Index (for images)."""
        try:
            from skimage.metrics import structural_similarity
            value = structural_similarity(original, reconstructed, data_range=original.max() - original.min())
        except ImportError:
            # Fallback to simple correlation
            value = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
        return MetricResult(value, "SSIM", "", True)
    
    @staticmethod
    def mae(original: np.ndarray, reconstructed: np.ndarray) -> MetricResult:
        """Mean Absolute Error."""
        value = np.mean(np.abs(original - reconstructed))
        return MetricResult(value, "MAE", "", False)
    
    @staticmethod
    def correlation(original: np.ndarray, reconstructed: np.ndarray) -> MetricResult:
        """Correlation coefficient."""
        value = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
        return MetricResult(value, "Correlation", "", True)
    
    @classmethod
    def compute_all(cls, original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, MetricResult]:
        """
        Compute all available metrics.
        
        Parameters
        ----------
        original : np.ndarray
            Original signal/image
        reconstructed : np.ndarray
            Reconstructed signal/image
            
        Returns
        -------
        Dict[str, MetricResult]
            Dictionary of all metrics
        """
        metrics = {
            'mse': cls.mse(original, reconstructed),
            'rmse': cls.rmse(original, reconstructed),
            'psnr': cls.psnr(original, reconstructed),
            'snr': cls.snr(original, reconstructed),
            'mae': cls.mae(original, reconstructed),
            'correlation': cls.correlation(original, reconstructed)
        }
        
        # Add SSIM for 2D images
        if original.ndim == 2:
            metrics['ssim'] = cls.ssim(original, reconstructed)
        
        return metrics


class PerformanceAnalyzer:
    """Analyze computational performance of SCSA methods."""
    
    def __init__(self):
        self.results = []
    
    def benchmark(self, method, signal: np.ndarray, n_runs: int = 10, **kwargs) -> Dict:
        """
        Benchmark a method's performance.
        
        Parameters
        ----------
        method : callable
            Method to benchmark
        signal : np.ndarray
            Input signal
        n_runs : int
            Number of runs for averaging
        **kwargs
            Additional parameters for method
            
        Returns
        -------
        Dict
            Performance statistics
        """
        times = []
        memory_usage = []
        
        for _ in range(n_runs):
            # Measure time
            start_time = time.perf_counter()
            result = method(signal, **kwargs)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
            # Estimate memory usage (simplified)
            if hasattr(result, 'reconstructed'):
                mem = result.reconstructed.nbytes
            else:
                mem = signal.nbytes
            memory_usage.append(mem)
        
        stats = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'mean_memory': np.mean(memory_usage),
            'signal_size': signal.size,
            'signal_shape': signal.shape,
            'method_name': method.__name__ if hasattr(method, '__name__') else str(method)
        }
        
        self.results.append(stats)
        return stats
    
    def compare_methods(self, methods: List, signal: np.ndarray, 
                       n_runs: int = 10) -> Dict:
        """
        Compare multiple methods.
        
        Parameters
        ----------
        methods : List
            List of (method, kwargs) tuples
        signal : np.ndarray
            Input signal
        n_runs : int
            Number of runs per method
            
        Returns
        -------
        Dict
            Comparison results
        """
        comparison = {}
        
        for method, kwargs in methods:
            name = method.__name__ if hasattr(method, '__name__') else str(method)
            comparison[name] = self.benchmark(method, signal, n_runs, **kwargs)
        
        return comparison
    
    def plot_comparison(self, comparison: Dict):
        """
        Plot performance comparison.
        
        Parameters
        ----------
        comparison : Dict
            Results from compare_methods
        """
        import matplotlib.pyplot as plt
        
        methods = list(comparison.keys())
        times = [comparison[m]['mean_time'] for m in methods]
        errors = [comparison[m]['std_time'] for m in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time comparison
        ax1.bar(methods, times, yerr=errors, capsize=5)
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory comparison
        memory = [comparison[m]['mean_memory'] / 1e6 for m in methods]  # Convert to MB
        ax2.bar(methods, memory)
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title('Memory Usage Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig