"""
Unit tests for visualization module.
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pyscsa.visualization import SCSAVisualizer
from pyscsa import SCSA1D, SCSA2D
from pyscsa.utils import SignalGenerator, add_noise

# Use non-interactive backend for testing
matplotlib.use('Agg')


class TestSCSAVisualizer:
    """Test suite for SCSAVisualizer class."""
    
    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance."""
        return SCSAVisualizer(figsize=(10, 6))
    
    @pytest.fixture
    def test_signal(self):
        """Generate test 1D signal."""
        x = np.linspace(-10, 10, 100)
        signal = SignalGenerator.sech_squared(x)
        noisy = add_noise(signal, snr_db=20, seed=42)
        
        # Apply SCSA
        scsa = SCSA1D(gmma=0.5)
        result = scsa.reconstruct(np.abs(noisy), h=1.0)
        
        return {
            'x': x,
            'original': signal,
            'noisy': noisy,
            'reconstructed': result.reconstructed,
            'result': result
        }
    
    @pytest.fixture
    def test_image(self):
        """Generate test 2D image."""
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        image = np.exp(-(X**2 + Y**2) / 4)
        noisy = add_noise(image, snr_db=15, seed=42)
        
        # Apply SCSA
        scsa = SCSA2D(gmma=2.0)
        result = scsa.reconstruct(image, h=5.0)
        
        return {
            'original': image,
            'noisy': noisy,
            'reconstructed': result.reconstructed,
            'result': result
        }
    
    def test_initialization(self):
        """Test visualizer initialization."""
        viz = SCSAVisualizer(style='default', figsize=(12, 8))
        assert viz.figsize == (12, 8)
        assert viz.style == 'default'
        
        # Test with seaborn style if available
        try:
            viz_seaborn = SCSAVisualizer(style='seaborn')
            assert viz_seaborn.style == 'seaborn'
        except:
            pass  # Seaborn might not be installed
    
    def test_plot_1d_comparison(self, visualizer, test_signal):
        """Test 1D comparison plot."""
        fig = visualizer.plot_1d_comparison(
            test_signal['original'],
            test_signal['reconstructed'],
            test_signal['noisy'],
            title="Test Plot",
            x_axis=test_signal['x'],
            metrics={'mse': 0.01, 'psnr': 20.0}
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check that axes were created
        axes = fig.get_axes()
        assert len(axes) >= 2  # At least main plot and error plot
        
        plt.close(fig)
    
    def test_plot_1d_comparison_without_noisy(self, visualizer, test_signal):
        """Test 1D comparison plot without noisy signal."""
        fig = visualizer.plot_1d_comparison(
            test_signal['original'],
            test_signal['reconstructed'],
            noisy=None,
            title="Test Plot Without Noisy"
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_2d_comparison(self, visualizer, test_image):
        """Test 2D comparison plot."""
        fig = visualizer.plot_2d_comparison(
            test_image['original'],
            test_image['reconstructed'],
            test_image['noisy'],
            title="Test 2D Plot",
            cmap='gray',
            metrics={'mse': 0.01, 'psnr': 20.0, 'ssim': 0.9}
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check that multiple subplots were created
        axes = fig.get_axes()
        assert len(axes) >= 6  # Should have multiple views
        
        plt.close(fig)
    
    def test_plot_2d_comparison_without_noisy(self, visualizer, test_image):
        """Test 2D comparison plot without noisy image."""
        fig = visualizer.plot_2d_comparison(
            test_image['original'],
            test_image['reconstructed'],
            noisy=None,
            title="Test 2D Plot Without Noisy"
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_eigenvalues(self, visualizer, test_signal):
        """Test eigenvalue spectrum plot."""
        # Get eigenvalues from result
        eigenvalues = test_signal['result'].eigenvalues
        
        fig = visualizer.plot_eigenvalues(
            eigenvalues,
            title="Test Eigenvalue Spectrum"
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Should have 2 subplots (linear and log scale)
        axes = fig.get_axes()
        assert len(axes) == 2
        
        plt.close(fig)
    
    def test_plot_eigenvalues_with_negative_values(self, visualizer):
        """Test eigenvalue plot with negative values (log scale should handle)."""
        eigenvalues = np.array([-1, 0, 1, 2, 3])
        
        fig = visualizer.plot_eigenvalues(eigenvalues)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_eigenfunctions(self, visualizer, test_signal):
        """Test eigenfunction plot."""
        # Get eigenfunctions from result
        eigenfunctions = test_signal['result'].eigenfunctions
        
        fig = visualizer.plot_eigenfunctions(
            eigenfunctions,
            n_functions=6,
            title="Test Eigenfunctions"
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_plot_eigenfunctions_limited(self, visualizer):
        """Test eigenfunction plot with limited functions."""
        # Create dummy eigenfunctions
        eigenfunctions = np.random.randn(100, 3)  # Only 3 functions
        
        fig = visualizer.plot_eigenfunctions(
            eigenfunctions,
            n_functions=10  # Request more than available
        )
        
        assert fig is not None
        # Should only plot 3 functions
        axes = fig.get_axes()
        assert len([ax for ax in axes if ax.has_data()]) <= 3
        
        plt.close(fig)
    
    def test_plot_parameter_sweep(self, visualizer):
        """Test parameter sweep plot."""
        param_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        metrics = {
            'MSE': [0.1, 0.05, 0.02, 0.03, 0.08],
            'PSNR': [10, 15, 20, 18, 12]
        }
        
        fig = visualizer.plot_parameter_sweep(
            param_values,
            metrics,
            param_name="gamma",
            optimal_value=1.0
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Should have one subplot per metric
        axes = fig.get_axes()
        assert len(axes) == len(metrics)
        
        plt.close(fig)
    
    def test_plot_convergence(self, visualizer):
        """Test convergence plot."""
        iterations = list(range(1, 11))
        values = [100, 50, 25, 15, 10, 8, 6, 5, 4.5, 4]
        
        fig = visualizer.plot_convergence(
            iterations,
            values,
            title="Test Convergence"
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_plot_metrics_comparison(self, visualizer):
        """Test metrics comparison plot."""
        metrics_dict = {
            'Method1': {'mse': 0.01, 'psnr': 20, 'ssim': 0.9},
            'Method2': {'mse': 0.02, 'psnr': 18, 'ssim': 0.85},
            'Method3': {'mse': 0.005, 'psnr': 22, 'ssim': 0.95}
        }
        
        fig = visualizer.plot_metrics_comparison(
            metrics_dict,
            title="Test Method Comparison"
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Should have one subplot per metric type
        axes = fig.get_axes()
        assert len(axes) == 3  # mse, psnr, ssim
        
        plt.close(fig)
    
    def test_plot_metrics_comparison_empty(self, visualizer):
        """Test metrics comparison with empty data."""
        metrics_dict = {}
        
        # Should handle empty dict gracefully
        result = visualizer.plot_metrics_comparison(metrics_dict)
        
        # Should return None or handle gracefully
        assert result is None or isinstance(result, plt.Figure)
        
        if result is not None:
            plt.close(result)
    
    def test_save_figure(self, visualizer, tmp_path, test_signal):
        """Test figure saving functionality."""
        # Create a simple plot
        fig = visualizer.plot_1d_comparison(
            test_signal['original'],
            test_signal['reconstructed']
        )
        
        # Save to temporary file
        filename = tmp_path / "test_figure.png"
        visualizer.save_figure(fig, str(filename), dpi=100, format='png')
        
        # Check file exists
        assert filename.exists()
        assert filename.stat().st_size > 0
        
        plt.close(fig)
    
    def test_save_figure_different_formats(self, visualizer, tmp_path):
        """Test saving in different formats."""
        # Create a simple figure
        fig = plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        
        formats = ['png', 'pdf', 'svg']
        for fmt in formats:
            filename = tmp_path / f"test_figure.{fmt}"
            visualizer.save_figure(fig, str(filename), format=fmt)
            assert filename.exists()
        
        plt.close(fig)
    
    def test_style_setting(self, visualizer):
        """Test style setting functionality."""
        # Test setting different styles
        styles_to_test = ['default', 'ggplot']
        
        for style in styles_to_test:
            if style in plt.style.available:
                visualizer.set_style(style)
                assert visualizer.style == style
    
    @pytest.mark.parametrize("figsize", [(8, 6), (12, 8), (16, 10)])
    def test_different_figsizes(self, figsize):
        """Test visualizer with different figure sizes."""
        viz = SCSAVisualizer(figsize=figsize)
        assert viz.figsize == figsize
        
        # Create a test plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        fig = viz.plot_1d_comparison(y, y * 0.9)
        assert fig.get_size_inches()[0] == figsize[0]
        assert fig.get_size_inches()[1] == figsize[1]
        
        plt.close(fig)
    
    def test_plot_with_nan_values(self, visualizer):
        """Test plots with NaN values."""
        # Create data with NaN
        original = np.array([1, 2, np.nan, 4, 5])
        reconstructed = np.array([1, 2, 3, 4, 5])
        
        # Should handle NaN gracefully
        fig = visualizer.plot_1d_comparison(original, reconstructed)
        assert fig is not None
        
        plt.close(fig)
    
    def test_plot_with_inf_values(self, visualizer):
        """Test plots with infinite values."""
        # Create data with inf
        original = np.array([1, 2, np.inf, 4, 5])
        reconstructed = np.array([1, 2, 3, 4, 5])
        
        # Should handle inf gracefully
        fig = visualizer.plot_1d_comparison(original, reconstructed)
        assert fig is not None
        
        plt.close(fig)
    
    def test_colormap_options(self, visualizer, test_image):
        """Test different colormap options for 2D plots."""
        colormaps = ['gray', 'viridis', 'plasma', 'hot']
        
        for cmap in colormaps:
            fig = visualizer.plot_2d_comparison(
                test_image['original'],
                test_image['reconstructed'],
                cmap=cmap
            )
            assert fig is not None
            plt.close(fig)


class TestVisualizationIntegration:
    """Integration tests for visualization with SCSA results."""
    
    def test_full_1d_workflow(self):
        """Test complete 1D visualization workflow."""
        # Generate signal
        x = np.linspace(-10, 10, 200)
        signal = SignalGenerator.double_well(x, separation=3, depth=50)
        noisy = add_noise(signal, snr_db=15, seed=42)
        
        # Apply SCSA
        scsa = SCSA1D(gmma=0.5)
        result = scsa.filter_with_optimal_h(np.abs(noisy))
        
        # Create visualizations
        viz = SCSAVisualizer()
        
        # 1D comparison
        fig1 = viz.plot_1d_comparison(
            np.abs(signal),
            result.reconstructed,
            np.abs(noisy),
            x_axis=x,
            metrics=result.metrics
        )
        assert fig1 is not None
        plt.close(fig1)
        
        # Eigenvalues
        fig2 = viz.plot_eigenvalues(result.eigenvalues)
        assert fig2 is not None
        plt.close(fig2)
        
        # Eigenfunctions
        fig3 = viz.plot_eigenfunctions(result.eigenfunctions, n_functions=4)
        assert fig3 is not None
        plt.close(fig3)
    
    def test_full_2d_workflow(self):
        """Test complete 2D visualization workflow."""
        # Generate image
        size = 50
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        image = np.exp(-(X**2 + Y**2) / 4)
        noisy = add_noise(image, snr_db=12, seed=42)
        
        # Apply SCSA
        scsa = SCSA2D(gmma=2.0)
        denoised = scsa.denoise(np.abs(noisy), method='windowed', 
                               window_size=10, h=5.0)
        
        # Create visualizations
        viz = SCSAVisualizer()
        
        # 2D comparison
        fig = viz.plot_2d_comparison(
            image,
            denoised,
            noisy,
            title="2D SCSA Results"
        )
        assert fig is not None
        plt.close(fig)
    
    def test_parameter_optimization_visualization(self):
        """Test visualization of parameter optimization."""
        x = np.linspace(-10, 10, 100)
        signal = SignalGenerator.sech_squared(x)
        noisy = add_noise(signal, snr_db=20, seed=42)
        
        # Test different gamma values
        gammas = [0.1, 0.5, 1.0, 2.0]
        mse_values = []
        psnr_values = []
        
        for gmma in gammas:
            scsa = SCSA1D(gmma=gmma)
            result = scsa.reconstruct(np.abs(noisy), h=1.0)
            mse_values.append(result.metrics['mse'])
            psnr_values.append(result.metrics['psnr'])
        
        # Visualize parameter sweep
        viz = SCSAVisualizer()
        fig = viz.plot_parameter_sweep(
            np.array(gammas),
            {'MSE': mse_values, 'PSNR': psnr_values},
            param_name='gamma',
            optimal_value=0.5
        )
        
        assert fig is not None
        plt.close(fig)
