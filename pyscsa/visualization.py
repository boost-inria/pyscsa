
"""
Visualization utilities for pyscsa library.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from typing import Optional, Tuple, List, Union, Dict
import warnings

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class SCSAVisualizer:
    """Enhanced visualization utilities for SCSA results."""
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize visualizer with style preferences.
        
        Parameters
        ----------
        style : str
            Matplotlib style ('default', 'seaborn', 'ggplot', etc.)
        figsize : Tuple[int, int]
            Default figure size
        """
        self.figsize = figsize
        self.set_style(style)
    
    def set_style(self, style: str):
        """Set plotting style."""
        if style == 'seaborn' and HAS_SEABORN:
            sns.set_style("whitegrid")
            sns.set_palette("husl")
        elif style in plt.style.available:
            plt.style.use(style)
        self.style = style
    
    def plot_1d_comparison(self, original: np.ndarray, reconstructed: np.ndarray,
                          noisy: Optional[np.ndarray] = None,
                          title: str = "SCSA 1D Reconstruction",
                          x_axis: Optional[np.ndarray] = None,
                          metrics: Optional[Dict] = None) -> Figure:
        """
        Enhanced 1D signal comparison plot.
        
        Parameters
        ----------
        original : np.ndarray
            Original signal
        reconstructed : np.ndarray
            Reconstructed signal
        noisy : np.ndarray, optional
            Noisy signal (if available)
        title : str
            Plot title
        x_axis : np.ndarray, optional
            X-axis values
        metrics : Dict, optional
            Metrics to display
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        if noisy is not None:
            fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] + 2))
            gs = GridSpec(3, 2, height_ratios=[2, 2, 1], figure=fig)
        else:
            fig = plt.figure(figsize=self.figsize)
            gs = GridSpec(2, 2, figure=fig)
        
        if x_axis is None:
            x_axis = np.arange(len(original))
        
        # Main comparison plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(x_axis, original, 'k-', label='Original', linewidth=2, alpha=0.8)
        if noisy is not None:
            ax1.plot(x_axis, noisy, 'r-', label='Noisy', linewidth=0.5, alpha=0.5)
        ax1.plot(x_axis, reconstructed, 'b--', label='SCSA', linewidth=1.5)
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Amplitude")
        ax1.set_title(title)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Error plot
        ax2 = fig.add_subplot(gs[1, 0])
        error = np.abs(original - reconstructed)
        ax2.plot(x_axis, error, 'g-', linewidth=1)
        ax2.fill_between(x_axis, 0, error, alpha=0.3, color='green')
        ax2.set_xlabel("Position")
        ax2.set_ylabel("Absolute Error")
        ax2.set_title("Reconstruction Error")
        ax2.grid(True, alpha=0.3)
        
        # Relative error plot
        ax3 = fig.add_subplot(gs[1, 1])
        relative_error = error / (np.abs(original) + 1e-10) * 100
        ax3.plot(x_axis, relative_error, 'orange', linewidth=1)
        ax3.set_xlabel("Position")
        ax3.set_ylabel("Relative Error (%)")
        ax3.set_title("Relative Error")
        ax3.grid(True, alpha=0.3)
        
        # Metrics display (if provided)
        if metrics is not None:
            if noisy is not None:
                ax4 = fig.add_subplot(gs[2, :])
            else:
                ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2, fig=fig)
            
            ax4.axis('off')
            metrics_text = "Performance Metrics:\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    metrics_text += f"{key.upper()}: {value:.4f}  "
            ax4.text(0.5, 0.5, metrics_text, ha='center', va='center',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_2d_comparison(self, original: np.ndarray, reconstructed: np.ndarray,
                          noisy: Optional[np.ndarray] = None,
                          title: str = "SCSA 2D Reconstruction",
                          cmap: str = 'gray',
                          metrics: Optional[Dict] = None) -> Figure:
        """
        Enhanced 2D image comparison plot.
        
        Parameters
        ----------
        original : np.ndarray
            Original image
        reconstructed : np.ndarray
            Reconstructed image
        noisy : np.ndarray, optional
            Noisy image
        title : str
            Plot title
        cmap : str
            Colormap
        metrics : Dict, optional
            Metrics to display
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        n_images = 3 if noisy is None else 4
        fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
        
        # Original image
        im0 = axes[0, 0].imshow(original, cmap=cmap)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
        
        # Noisy image (if provided)
        col_idx = 1
        if noisy is not None:
            im_noisy = axes[0, col_idx].imshow(noisy, cmap=cmap)
            axes[0, col_idx].set_title("Noisy")
            axes[0, col_idx].axis('off')
            plt.colorbar(im_noisy, ax=axes[0, col_idx], fraction=0.046)
            col_idx += 1
        
        # Reconstructed image
        im1 = axes[0, col_idx].imshow(reconstructed, cmap=cmap)
        axes[0, col_idx].set_title("SCSA Reconstructed")
        axes[0, col_idx].axis('off')
        plt.colorbar(im1, ax=axes[0, col_idx], fraction=0.046)
        
        # Difference image
        diff = np.abs(original - reconstructed)
        im2 = axes[0, col_idx + 1].imshow(diff, cmap='hot')
        axes[0, col_idx + 1].set_title("Absolute Difference")
        axes[0, col_idx + 1].axis('off')
        plt.colorbar(im2, ax=axes[0, col_idx + 1], fraction=0.046)
        
        # Histogram comparison
        axes[1, 0].hist(original.flatten(), bins=50, alpha=0.7, label='Original', density=True)
        axes[1, 0].hist(reconstructed.flatten(), bins=50, alpha=0.7, label='Reconstructed', density=True)
        axes[1, 0].set_xlabel("Intensity")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title("Intensity Distribution")
        axes[1, 0].legend()
        
        # Error distribution
        axes[1, 1].hist(diff.flatten(), bins=50, color='red', alpha=0.7)
        axes[1, 1].set_xlabel("Error Magnitude")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Error Distribution")
        
        # Cross-section comparison
        mid_row = original.shape[0] // 2
        axes[1, 2].plot(original[mid_row, :], 'k-', label='Original', linewidth=2)
        axes[1, 2].plot(reconstructed[mid_row, :], 'b--', label='Reconstructed', linewidth=1.5)
        axes[1, 2].set_xlabel("Column Index")
        axes[1, 2].set_ylabel("Intensity")
        axes[1, 2].set_title(f"Cross-section at Row {mid_row}")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Metrics display
        if n_images == 4:
            axes[1, 3].axis('off')
            if metrics:
                metrics_text = "Metrics:\n\n"
                for key, value in metrics.items():
                    if isinstance(value, float):
                        metrics_text += f"{key.upper()}: {value:.4f}\n"
                axes[1, 3].text(0.5, 0.5, metrics_text, ha='center', va='center',
                              fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig
    
    def plot_eigenvalues(self, eigenvalues: np.ndarray, title: str = "SCSA Eigenvalue Spectrum") -> Figure:
        """
        Plot eigenvalue spectrum.
        
        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues from SCSA
        title : str
            Plot title
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Extract diagonal if matrix
        if eigenvalues.ndim == 2:
            eigenvals = np.diag(eigenvalues)
        else:
            eigenvals = eigenvalues
        
        # Linear scale
        axes[0].plot(eigenvals, 'o-', markersize=4)
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("Eigenvalue")
        axes[0].set_title("Eigenvalue Spectrum (Linear)")
        axes[0].grid(True, alpha=0.3)
        
        # Log scale
        if np.all(eigenvals > 0):
            axes[1].semilogy(eigenvals, 'o-', markersize=4, color='red')
            axes[1].set_xlabel("Index")
            axes[1].set_ylabel("Eigenvalue (log scale)")
            axes[1].set_title("Eigenvalue Spectrum (Log)")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "Cannot plot log scale\n(non-positive values)",
                        ha='center', va='center')
            axes[1].set_title("Eigenvalue Spectrum (Log)")
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_eigenfunctions(self, eigenfunctions: np.ndarray, 
                           n_functions: int = 6,
                           title: str = "SCSA Eigenfunctions") -> Figure:
        """
        Plot first n eigenfunctions.
        
        Parameters
        ----------
        eigenfunctions : np.ndarray
            Eigenfunctions matrix
        n_functions : int
            Number of functions to plot
        title : str
            Plot title
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        n_to_plot = min(n_functions, eigenfunctions.shape[1])
        n_rows = (n_to_plot + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_to_plot == 1 else axes
        
        for i in range(n_to_plot):
            axes[i].plot(eigenfunctions[:, i])
            axes[i].set_title(f"Eigenfunction {i+1}")
            axes[i].set_xlabel("Position")
            axes[i].set_ylabel("Amplitude")
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_to_plot, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_parameter_sweep(self, param_values: np.ndarray, 
                           metrics: Dict[str, List[float]],
                           param_name: str = "h",
                           optimal_value: Optional[float] = None) -> Figure:
        """
        Plot parameter sweep results.
        
        Parameters
        ----------
        param_values : np.ndarray
            Parameter values tested
        metrics : Dict[str, List[float]]
            Metrics for each parameter value
        param_name : str
            Name of parameter
        optimal_value : float, optional
            Optimal parameter value to highlight
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            axes[idx].plot(param_values, values, 'o-', markersize=4)
            
            if optimal_value is not None:
                # Find closest index
                opt_idx = np.argmin(np.abs(param_values - optimal_value))
                axes[idx].plot(optimal_value, values[opt_idx], 'r*', 
                             markersize=15, label=f'Optimal {param_name}={optimal_value:.3f}')
                axes[idx].legend()
            
            axes[idx].set_xlabel(f"Parameter {param_name}")
            axes[idx].set_ylabel(metric_name.upper())
            axes[idx].set_title(f"{metric_name.upper()} vs {param_name}")
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle(f"Parameter Sweep: {param_name}")
        plt.tight_layout()
        return fig
    
    def plot_convergence(self, iterations: List[int], values: List[float],
                        title: str = "SCSA Convergence") -> Figure:
        """
        Plot convergence history.
        
        Parameters
        ----------
        iterations : List[int]
            Iteration numbers
        values : List[float]
            Convergence values
        title : str
            Plot title
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(iterations, values, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost Function Value")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add convergence rate annotation
        if len(values) > 1:
            conv_rate = (values[-1] - values[0]) / len(values)
            ax.text(0.7, 0.9, f"Avg. Rate: {conv_rate:.4e}/iter", 
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict[str, float]],
                               title: str = "Method Comparison") -> Figure:
        """
        Compare metrics across different methods.
        
        Parameters
        ----------
        metrics_dict : Dict[str, Dict[str, float]]
            Nested dict: {method_name: {metric_name: value}}
        title : str
            Plot title
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        methods = list(metrics_dict.keys())
        if not methods:
            warnings.warn("No methods to compare")
            return None
        
        metric_names = list(metrics_dict[methods[0]].keys())
        n_metrics = len(metric_names)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metric_names):
            values = [metrics_dict[method][metric] for method in methods]
            bars = axes[idx].bar(methods, values)
            
            # Color based on metric type (lower is better for errors)
            if metric.lower() in ['mse', 'rmse', 'mae', 'error']:
                colors = ['red' if v == max(values) else 'green' if v == min(values) else 'orange' 
                         for v in values]
            else:
                colors = ['green' if v == max(values) else 'red' if v == min(values) else 'orange' 
                         for v in values]
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            axes[idx].set_ylabel(metric.upper())
            axes[idx].set_title(f"{metric.upper()} Comparison")
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{val:.3f}', ha='center', va='bottom')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def save_figure(fig: Figure, filename: str, dpi: int = 300, 
                   format: str = 'png', **kwargs):
        """
        Save figure with high quality.
        
        Parameters
        ----------
        fig : Figure
            Matplotlib figure
        filename : str
            Output filename
        dpi : int
            Resolution
        format : str
            File format
        **kwargs
            Additional savefig parameters
        """
        fig.savefig(filename, dpi=dpi, format=format, bbox_inches='tight', **kwargs)
        print(f"Figure saved to {filename}")
