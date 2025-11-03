"""
Unit tests for SCSA1D class.
"""

import pytest
import numpy as np
from pyscsa import SCSA1D, add_noise
from pyscsa.utils import SignalGenerator


class TestSCSA1D:
    """Test suite for SCSA1D functionality."""
    
    @pytest.fixture
    def simple_signal(self):
        """Generate a simple test signal."""
        x = np.linspace(-10, 10, 100)
        return SignalGenerator.sech_squared(x)
    
    @pytest.fixture
    def noisy_signal(self, simple_signal):
        """Generate noisy version of test signal."""
        return add_noise(simple_signal, snr_db=20, seed=42)
    
    def test_initialization(self):
        """Test SCSA1D initialization."""
        scsa = SCSA1D(gmma=0.5)
        assert scsa._gmma == 0.5
        
        with pytest.raises(ValueError):
            SCSA1D(gmma=-1)  # Negative gamma should raise error
    
    def test_basic_reconstruction(self, simple_signal):
        """Test basic signal reconstruction."""
        scsa = SCSA1D(gmma=0.5)
        result = scsa.reconstruct(simple_signal, h=1.0)
        
        assert result.reconstructed is not None
        assert result.reconstructed.shape == simple_signal.shape
        assert result.num_eigenvalues > 0
    
    def test_optimal_h_filtering(self, noisy_signal):
        """Test filtering with optimal h selection."""
        scsa = SCSA1D(gmma=0.5)
        result = scsa.filter_with_c_scsa(noisy_signal)
        
        assert result.optimal_h is not None
        assert result.optimal_h > 0
        assert result.metrics is not None
        assert 'mse' in result.metrics
    
    def test_denoise_convenience(self, noisy_signal):
        """Test denoise convenience method."""
        scsa = SCSA1D()
        denoised = scsa.denoise(noisy_signal)
        
        assert denoised is not None
        assert denoised.shape == noisy_signal.shape
    
    def test_metrics_computation(self, simple_signal, noisy_signal):
        """Test metrics computation."""
        scsa = SCSA1D()
        result = scsa.reconstruct(noisy_signal)
        
        assert 'mse' in result.metrics
        assert 'psnr' in result.metrics
        assert result.metrics['mse'] >= 0
        assert result.metrics['psnr'] > 0
    
    @pytest.mark.parametrize("gmma", [0.1, 0.5, 1.0, 2.0])
    def test_different_gammas(self, simple_signal, gmma):
        """Test reconstruction with different gamma values."""
        scsa = SCSA1D(gmma=gmma)
        result = scsa.reconstruct(simple_signal)
        
        assert result.reconstructed is not None
        assert not np.isnan(result.reconstructed).any()
