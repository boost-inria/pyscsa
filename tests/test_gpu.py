"""
Unit tests for GPU-accelerated SCSA classes.
"""
import pytest
import numpy as np

# Try to import GPU classes
try:
    from pyscsa import SCSA1D_GPU, SCSA2D_GPU, CUPY_AVAILABLE
    import cupy as cp
    GPU_AVAILABLE = CUPY_AVAILABLE
except ImportError:
    GPU_AVAILABLE = False
    SCSA1D_GPU = None
    SCSA2D_GPU = None

from pyscsa import SCSA1D, SCSA2D, add_noise


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU/CuPy not available")
class TestSCSA1D_GPU:
    """Test suite for SCSA1D_GPU functionality."""
    
    @pytest.fixture
    def simple_signal(self):
        """Generate a simple test signal."""
        x = np.linspace(-10, 10, 500)
        return -2 * (1/np.cosh(x))**2
    
    @pytest.fixture
    def noisy_signal(self, simple_signal):
        """Generate noisy version of test signal."""
        return add_noise(simple_signal, snr_db=20, seed=42)
    
    def test_initialization(self):
        """Test SCSA1D_GPU initialization."""
        scsa_gpu = SCSA1D_GPU(gmma=0.5, device_id=0)
        assert scsa_gpu._gmma == 0.5
        assert scsa_gpu.device_id == 0
        
        with pytest.raises(ValueError):
            SCSA1D_GPU(gmma=0)  # Zero gamma should raise error
    
    def test_basic_reconstruction(self, simple_signal):
        """Test basic signal reconstruction on GPU."""
        scsa_gpu = SCSA1D_GPU(gmma=0.5)
        result = scsa_gpu.reconstruct(simple_signal, h=1.0)
        
        assert result.reconstructed is not None
        assert len(result.reconstructed) == len(simple_signal)
        assert result.num_eigenvalues > 0
        assert isinstance(result.reconstructed, np.ndarray)  # Should be on CPU
    
    def test_reconstruction_with_lambda(self, simple_signal):
        """Test reconstruction with custom lambda."""
        scsa_gpu = SCSA1D_GPU(gmma=0.5)
        result = scsa_gpu.reconstruct(simple_signal, h=1.0, lambda_g=-1.0)
        
        assert result.reconstructed is not None
        assert result.num_eigenvalues > 0
    
    def test_c_scsa_filtering(self, noisy_signal):
        """Test C-SCSA filtering with automatic h optimization on GPU."""
        scsa_gpu = SCSA1D_GPU(gmma=0.5)
        result = scsa_gpu.filter_with_c_scsa(noisy_signal, curvature_weight=4.0)
        
        assert result.reconstructed is not None
        assert hasattr(result, 'optimal_h')
        assert result.optimal_h > 0
        assert result.metrics is not None
    
    def test_denoise_convenience(self, noisy_signal):
        """Test denoise convenience method on GPU."""
        scsa_gpu = SCSA1D_GPU(gmma=0.5)
        denoised = scsa_gpu.denoise(noisy_signal)
        
        assert denoised is not None
        assert len(denoised) == len(noisy_signal)
        assert isinstance(denoised, np.ndarray)
    
    def test_metrics_computation(self, simple_signal, noisy_signal):
        """Test metrics computation for GPU."""
        scsa_gpu = SCSA1D_GPU(gmma=0.5)
        result = scsa_gpu.reconstruct(noisy_signal, h=1.0)
        
        assert result.metrics is not None
        assert 'mse' in result.metrics
        assert 'rmse' in result.metrics
        assert 'psnr' in result.metrics
        assert 'snr' in result.metrics
        assert result.metrics['mse'] >= 0
        assert result.metrics['psnr'] > 0
    
    def test_negative_signal_handling(self):
        """Test handling of signals with negative values on GPU."""
        signal = np.sin(np.linspace(0, 4*np.pi, 200))
        scsa_gpu = SCSA1D_GPU(gmma=0.5)
        result = scsa_gpu.reconstruct(signal, h=1.0)
        
        assert result.reconstructed is not None
        assert len(result.reconstructed) == len(signal)
    
    def test_gpu_vs_cpu_consistency(self, simple_signal):
        """Test that GPU results match CPU results."""
        scsa_cpu = SCSA1D(gmma=0.5)
        scsa_gpu = SCSA1D_GPU(gmma=0.5)
        
        result_cpu = scsa_cpu.reconstruct(simple_signal, h=1.0)
        result_gpu = scsa_gpu.reconstruct(simple_signal, h=1.0)
        
        # Results should be very similar (within numerical precision)
        np.testing.assert_allclose(
            result_cpu.reconstructed, 
            result_gpu.reconstructed, 
            rtol=1e-4, 
            atol=1e-6
        )
        assert result_cpu.num_eigenvalues == result_gpu.num_eigenvalues
    
    def test_large_signal_processing(self):
        """Test GPU processing of large signals."""
        x = np.linspace(-20, 20, 5000)
        signal = np.exp(-x**2/10) * np.cos(2*x)
        
        scsa_gpu = SCSA1D_GPU(gmma=0.5)
        result = scsa_gpu.reconstruct(signal, h=2.0)
        
        assert result.reconstructed is not None
        assert len(result.reconstructed) == len(signal)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU/CuPy not available")
class TestSCSA2D_GPU:
    """Test suite for SCSA2D_GPU functionality."""
    
    @pytest.fixture
    def simple_image(self):
        """Generate a simple test image."""
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        return np.exp(-(X**2 + Y**2) / 4)
    
    @pytest.fixture
    def noisy_image(self, simple_image):
        """Generate noisy version of test image."""
        return add_noise(simple_image, snr_db=15, seed=42)
    
    @pytest.fixture
    def small_image(self):
        """Generate a small test image for faster tests."""
        x = np.linspace(-3, 3, 20)
        y = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x, y)
        return np.exp(-(X**2 + Y**2) / 2)
    
    def test_initialization(self):
        """Test SCSA2D_GPU initialization."""
        scsa_gpu = SCSA2D_GPU(gmma=2.0, device_id=0)
        assert scsa_gpu._gmma == 2.0
        assert scsa_gpu.device_id == 0
        
        with pytest.raises(ValueError):
            SCSA2D_GPU(gmma=0)  # Zero gamma should raise error
    
    def test_basic_reconstruction(self, small_image):
        """Test basic image reconstruction on GPU."""
        scsa_gpu = SCSA2D_GPU(gmma=2.0)
        result = scsa_gpu.reconstruct(small_image, h=2.0)
        
        assert result.reconstructed is not None
        assert result.reconstructed.shape == small_image.shape
        assert result.num_eigenvalues > 0
        assert isinstance(result.reconstructed, np.ndarray)  # Should be on CPU
    
    def test_reconstruction_with_lambda(self, small_image):
        """Test reconstruction with custom lambda."""
        scsa_gpu = SCSA2D_GPU(gmma=2.0)
        result = scsa_gpu.reconstruct(small_image, h=2.0, lambda_g=-0.5)
        
        assert result.reconstructed is not None
        assert result.num_eigenvalues > 0
    
    def test_metrics_computation(self, small_image, noisy_image):
        """Test metrics computation for 2D GPU."""
        noisy_small = noisy_image[:20, :20]  # Use smaller subset
        
        scsa_gpu = SCSA2D_GPU(gmma=2.0)
        result = scsa_gpu.reconstruct(noisy_small, h=2.0)
        
        assert result.metrics is not None
        assert 'mse' in result.metrics
        assert 'psnr' in result.metrics
        assert 'snr' in result.metrics
        assert result.metrics['mse'] >= 0
    
    def test_negative_image_handling(self):
        """Test handling of images with negative values on GPU."""
        x = np.linspace(-3, 3, 30)
        y = np.linspace(-3, 3, 30)
        X, Y = np.meshgrid(x, y)
        image = np.sin(X) * np.cos(Y)
        
        scsa_gpu = SCSA2D_GPU(gmma=2.0)
        result = scsa_gpu.reconstruct(image, h=2.0)
        
        assert result.reconstructed is not None
        assert result.reconstructed.shape == image.shape
    
    def test_gpu_vs_cpu_consistency(self, small_image):
        """Test that GPU results match CPU results for 2D."""
        scsa_cpu = SCSA2D(gmma=2.0)
        scsa_gpu = SCSA2D_GPU(gmma=2.0)
        
        result_cpu = scsa_cpu.reconstruct(small_image, h=2.0)
        result_gpu = scsa_gpu.reconstruct(small_image, h=2.0)
        
        # Results should be very similar (within numerical precision)
        np.testing.assert_allclose(
            result_cpu.reconstructed, 
            result_gpu.reconstructed, 
            rtol=1e-3,  # Slightly relaxed for 2D
            atol=1e-5
        )
    
    def test_medium_image_processing(self):
        """Test GPU processing of medium-sized images."""
        x = np.linspace(-5, 5, 64)
        y = np.linspace(-5, 5, 64)
        X, Y = np.meshgrid(x, y)
        image = np.exp(-(X**2 + Y**2) / 4) * np.cos(X) * np.sin(Y)
        
        scsa_gpu = SCSA2D_GPU(gmma=2.0)
        result = scsa_gpu.reconstruct(image, h=3.0)
        
        assert result.reconstructed is not None
        assert result.reconstructed.shape == image.shape
    
    def test_different_gamma_values(self, small_image):
        """Test reconstruction with different gamma values on GPU."""
        for gamma in [1.0, 2.0, 3.0]:
            scsa_gpu = SCSA2D_GPU(gmma=gamma)
            result = scsa_gpu.reconstruct(small_image, h=2.0)
            
            assert result.reconstructed is not None
            assert result.reconstructed.shape == small_image.shape


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU/CuPy not available")
class TestGPUPerformance:
    """Test suite for GPU performance characteristics."""
    
    def test_gpu_memory_transfer(self):
        """Test GPU memory transfer operations."""
        from pyscsa.scsa_gpu import GPUMixin
        
        # Test CPU to GPU transfer
        cpu_array = np.random.rand(1000)
        gpu_array = GPUMixin._to_gpu(cpu_array)
        assert isinstance(gpu_array, cp.ndarray)
        
        # Test GPU to CPU transfer
        cpu_back = GPUMixin._to_cpu(gpu_array)
        assert isinstance(cpu_back, np.ndarray)
        np.testing.assert_array_equal(cpu_array, cpu_back)
    
    def test_multiple_gpu_calls(self):
        """Test multiple consecutive GPU operations."""
        x = np.linspace(-5, 5, 200)
        signal = np.exp(-x**2)
        
        scsa_gpu = SCSA1D_GPU(gmma=0.5)
        
        # Run multiple reconstructions
        for h in [0.5, 1.0, 1.5]:
            result = scsa_gpu.reconstruct(signal, h=h)
            assert result.reconstructed is not None
    
    def test_device_selection(self):
        """Test device selection for multi-GPU systems."""
        # This test will pass with single GPU (device_id=0)
        scsa_gpu = SCSA1D_GPU(gmma=0.5, device_id=0)
        
        x = np.linspace(-5, 5, 100)
        signal = np.exp(-x**2)
        result = scsa_gpu.reconstruct(signal, h=1.0)
        
        assert result.reconstructed is not None


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU/CuPy not available")
class TestGPUEdgeCases:
    """Test edge cases specific to GPU implementation."""
    
    def test_empty_eigenvalues(self):
        """Test handling when no eigenvalues below threshold."""
        # Create signal that might not have eigenvalues below threshold
        signal = np.ones(100) * 0.001  # Very small constant signal
        
        scsa_gpu = SCSA1D_GPU(gmma=0.5)
        result = scsa_gpu.reconstruct(signal, h=0.01, lambda_g=10.0)
        
        # Should handle gracefully
        assert result.reconstructed is not None
    
    def test_single_pixel_image(self):
        """Test 2D reconstruction with very small image."""
        image = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        scsa_gpu = SCSA2D_GPU(gmma=2.0)
        result = scsa_gpu.reconstruct(image, h=1.0)
        
        assert result.reconstructed is not None
        assert result.reconstructed.shape == image.shape
    
    def test_high_noise_signal(self):
        """Test GPU handling of very noisy signals."""
        x = np.linspace(-5, 5, 300)
        signal = np.exp(-x**2)
        noisy = add_noise(signal, snr_db=5, seed=42)  # High noise
        
        scsa_gpu = SCSA1D_GPU(gmma=0.5)
        result = scsa_gpu.reconstruct(noisy, h=1.0)
        
        assert result.reconstructed is not None
        assert result.metrics['snr'] < 20  # Should reflect high noise


class TestGPUAvailability:
    """Tests that run regardless of GPU availability."""
    
    def test_cupy_import_flag(self):
        """Test that CUPY_AVAILABLE flag is set correctly."""
        from pyscsa import CUPY_AVAILABLE
        assert isinstance(CUPY_AVAILABLE, bool)
    
    @pytest.mark.skipif(GPU_AVAILABLE, reason="Test only when GPU unavailable")
    def test_gpu_unavailable_error(self):
        """Test that appropriate error is raised when GPU unavailable."""
        with pytest.raises(RuntimeError, match="CuPy not available"):
            SCSA1D_GPU(gmma=0.5)
    
    @pytest.mark.skipif(GPU_AVAILABLE, reason="Test only when GPU unavailable")
    def test_gpu_2d_unavailable_error(self):
        """Test that appropriate error is raised for 2D when GPU unavailable."""
        with pytest.raises(RuntimeError, match="CuPy not available"):
            SCSA2D_GPU(gmma=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
