"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
from pyscsa.utils import (
    SignalGenerator, NoiseGenerator,
    compute_snr, compute_psnr,
    estimate_noise_level, adaptive_threshold
)


class TestSignalGenerator:
    """Test signal generation utilities."""
    
    def test_sech_squared(self):
        """Test sech-squared signal generation."""
        x = np.linspace(-10, 10, 100)
        signal = SignalGenerator.sech_squared(x, center=0, width=2)
        
        assert signal.shape == x.shape
        assert np.all(signal <= 0)  # Should be negative
        assert signal[50] == np.min(signal)  # Minimum at center
    
    def test_double_well(self):
        """Test double-well potential generation."""
        x = np.linspace(-10, 10, 100)
        signal = SignalGenerator.double_well(x, separation=3)
        
        assert signal.shape == x.shape
        assert np.sum(signal < -40) >= 2  # Should have two wells
    
    def test_gaussian_mixture(self):
        """Test Gaussian mixture generation."""
        x = np.linspace(0, 10, 100)
        centers = [2, 5, 8]
        amplitudes = [1, 2, 1.5]
        widths = [0.5, 0.3, 0.4]
        
        signal = SignalGenerator.gaussian_mixture(x, centers, amplitudes, widths)
        
        assert signal.shape == x.shape
        assert np.all(signal >= 0)  # Should be positive


class TestNoiseGenerator:
    """Test noise generation utilities."""
    
    def test_gaussian_noise(self):
        """Test Gaussian noise generation."""
        shape = (100,)
        noise = NoiseGenerator.gaussian(shape, snr_db=20, seed=42)
        
        assert noise.shape == shape
        assert np.abs(np.mean(noise)) < 0.1  # Should be zero-mean
    
    def test_salt_pepper_noise(self):
        """Test salt and pepper noise generation."""
        shape = (50, 50)
        noise = NoiseGenerator.salt_pepper(shape, prob=0.1, seed=42)
        
        assert noise.shape == shape
        assert set(np.unique(noise)).issubset({-1, 0, 1})


class TestMetrics:
    """Test metric computation functions."""
    
    def test_compute_snr(self):
        """Test SNR computation."""
        signal = np.ones(100)
        noise = 0.1 * np.random.randn(100)
        
        snr = compute_snr(signal, noise)
        
        assert snr > 0
        assert snr < 50  # Reasonable range
    
    def test_compute_psnr(self):
        """Test PSNR computation."""
        original = np.ones(100)
        reconstructed = original + 0.1 * np.random.randn(100)
        
        psnr = compute_psnr(original, reconstructed)
        
        assert psnr > 0
        assert psnr < 100  # Reasonable range
    
    def test_estimate_noise_level(self):
        """Test noise level estimation."""
        np.random.seed(42)
        clean = np.sin(np.linspace(0, 10, 100))
        noise_level = 0.1
        noisy = clean + noise_level * np.random.randn(100)
        
        estimated = estimate_noise_level(noisy, method='mad')
        
        assert estimated > 0
        assert estimated < 1.0
