"""
Unit tests for SCSA2D class.
"""

import pytest
import numpy as np
from pyscsa import SCSA2D, add_noise


class TestSCSA2D:
    """Test suite for SCSA2D functionality."""
    
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
    
    def test_initialization(self):
        """Test SCSA2D initialization."""
        scsa = SCSA2D(gmma=2.0)
        assert scsa._gmma == 2.0
        
        with pytest.raises(ValueError):
            SCSA2D(gmma=0)  # Zero gamma should raise error
    
    def test_basic_reconstruction(self, simple_image):
        """Test basic image reconstruction."""
        scsa = SCSA2D(gmma=2.0)
        result = scsa.reconstruct(simple_image, h=5.0)
        
        assert result.reconstructed is not None
        assert result.reconstructed.shape == simple_image.shape
        assert result.num_eigenvalues > 0
    
    def test_windowed_reconstruction(self, simple_image):
        """Test windowed reconstruction."""
        scsa = SCSA2D(gmma=2.0)
        reconstructed = scsa.reconstruct_windowed(
            simple_image, h=5.0, window_size=10, stride=5
        )
        
        assert reconstructed is not None
        assert reconstructed.shape == simple_image.shape
    
    def test_denoise_methods(self, noisy_image):
        """Test different denoising methods."""
        scsa = SCSA2D()
        
        # Test standard method
        denoised_standard = scsa.denoise(
            np.abs(noisy_image), method='standard', h=5.0
        )
        assert denoised_standard.shape == noisy_image.shape
        
        # Test windowed method
        denoised_windowed = scsa.denoise(
            np.abs(noisy_image), method='windowed', 
            h=5.0, window_size=10
        )
        assert denoised_windowed.shape == noisy_image.shape
    
    def test_metrics_computation(self, simple_image, noisy_image):
        """Test metrics computation for 2D."""
        scsa = SCSA2D()
        result = scsa.reconstruct(np.abs(noisy_image), h=5.0)
        
        assert result.metrics is not None
        assert 'mse' in result.metrics
        assert 'psnr' in result.metrics
