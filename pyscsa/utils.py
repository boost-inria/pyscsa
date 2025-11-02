"""
Utility functions and classes for SCSA library.
"""

import numpy as np
from typing import Optional, Tuple, Union
from scipy import signal
from scipy.stats import norm


class SignalGenerator:
    """Generate various test signals for SCSA testing."""
    
    @staticmethod
    def sech_squared(x: np.ndarray, center: float = 0, width: float = 1) -> np.ndarray:
        """Generate sech-squared potential (quantum well)."""
        return -2 * (1/np.cosh((x - center) / width))**2
    
    @staticmethod
    def double_well(x: np.ndarray, separation: float = 3, 
                   depth: float = 50, width: float = 1) -> np.ndarray:
        """Generate double-well potential."""
        well1 = -depth * np.exp(-((x - separation)**2) / (2 * width**2))
        well2 = -depth * np.exp(-((x + separation)**2) / (2 * width**2))
        return well1 + well2
    
    @staticmethod
    def gaussian_mixture(x: np.ndarray, centers: list, 
                        amplitudes: list, widths: list) -> np.ndarray:
        """Generate mixture of Gaussian peaks."""
        signal = np.zeros_like(x)
        for c, a, w in zip(centers, amplitudes, widths):
            signal += a * np.exp(-((x - c)**2) / (2 * w**2))
        return signal
    
    @staticmethod
    def chirp(t: np.ndarray, f0: float = 1, f1: float = 10, 
             method: str = 'linear') -> np.ndarray:
        """Generate chirp signal."""
        return signal.chirp(t, f0, t[-1], f1, method=method)
    
    @staticmethod
    def step_function(x: np.ndarray, steps: list) -> np.ndarray:
        """Generate piecewise constant signal."""
        signal_out = np.zeros_like(x)
        for i, (pos, height) in enumerate(steps):
            if i == 0:
                mask = x <= pos
            elif i == len(steps) - 1:
                mask = x > steps[i-1][0]
            else:
                mask = (x > steps[i-1][0]) & (x <= pos)
            signal_out[mask] = height
        return signal_out


class NoiseGenerator:
    """Generate various types of noise for testing."""
    
    @staticmethod
    def gaussian(shape: Union[int, Tuple], snr_db: float, 
                signal_power: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
        """Generate Gaussian white noise."""
        if seed is not None:
            np.random.seed(seed)
        
        noise_power = signal_power * 10**(-snr_db / 10)
        return np.random.normal(0, np.sqrt(noise_power), shape)
    
    @staticmethod
    def poisson(shape: Union[int, Tuple], lam: float = 1.0, 
               seed: Optional[int] = None) -> np.ndarray:
        """Generate Poisson noise."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.poisson(lam, shape)
    
    @staticmethod
    def salt_pepper(shape: Union[int, Tuple], prob: float = 0.05,
                   seed: Optional[int] = None) -> np.ndarray:
        """Generate salt and pepper noise."""
        if seed is not None:
            np.random.seed(seed)
        
        noise = np.zeros(shape)
        # Salt noise
        num_salt = int(prob * np.prod(shape) * 0.5)
        coords_salt = [np.random.randint(0, i, num_salt) for i in shape]
        noise[tuple(coords_salt)] = 1
        
        # Pepper noise
        num_pepper = int(prob * np.prod(shape) * 0.5)
        coords_pepper = [np.random.randint(0, i, num_pepper) for i in shape]
        noise[tuple(coords_pepper)] = -1
        
        return noise
    
    @staticmethod
    def uniform(shape: Union[int, Tuple], low: float = -0.5, 
               high: float = 0.5, seed: Optional[int] = None) -> np.ndarray:
        """Generate uniform noise."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(low, high, shape)


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio.
    
    Parameters
    ----------
    signal : np.ndarray
        Clean signal
    noise : np.ndarray
        Noise component
        
    Returns
    -------
    float
        SNR in dB
    """
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray, 
                max_val: Optional[float] = None) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Parameters
    ----------
    original : np.ndarray
        Original signal
    reconstructed : np.ndarray
        Reconstructed signal
    max_val : float, optional
        Maximum possible value. If None, use max of original
        
    Returns
    -------
    float
        PSNR in dB
    """
    mse = np.mean((original - reconstructed)**2)
    
    if mse == 0:
        return float('inf')
    
    if max_val is None:
        max_val = np.max(original)
    
    return 20 * np.log10(max_val / np.sqrt(mse))


def estimate_noise_level(signal: np.ndarray, method: str = 'mad') -> float:
    """
    Estimate noise level in a signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    method : str
        Estimation method ('mad', 'std', 'wavelet')
        
    Returns
    -------
    float
        Estimated noise standard deviation
    """
    if method == 'mad':
        # Median Absolute Deviation
        from scipy.stats import median_abs_deviation
        return 1.4826 * median_abs_deviation(signal, scale=1.0)
    elif method == 'std':
        # High-pass filter then standard deviation
        if signal.ndim == 1:
            hp_signal = np.diff(signal)
        else:
            hp_signal = np.gradient(signal)[0]
        return np.std(hp_signal) / np.sqrt(2)
    elif method == 'wavelet':
        # Wavelet-based estimation
        import pywt
        coeffs = pywt.dwt(signal, 'db1')
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        return sigma
    else:
        raise ValueError(f"Unknown method: {method}")

def add_noise(signal: np.ndarray, snr_db: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Add Gaussian white noise to a signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    snr_db : float
        Desired SNR in dB
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Noisy signal
    """
    if seed is not None:
        np.random.seed(seed)
    
    signal_power = np.mean(signal**2)
    noise_power = signal_power * 10**(-snr_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    
    return signal + noise


def normalize_signal(signal: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize a signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    method : str, default='minmax'
        Normalization method ('minmax' or 'zscore')
        
    Returns
    -------
    np.ndarray
        Normalized signal
    """
    if method == 'minmax':
        return (signal - signal.min()) / (signal.max() - signal.min() + 1e-10)
    elif method == 'zscore':
        return (signal - signal.mean()) / (signal.std() + 1e-10)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
        
def adaptive_threshold(signal: np.ndarray, method: str = 'otsu') -> float:
    """
    Compute adaptive threshold for signal processing.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    method : str
        Thresholding method ('otsu', 'mean', 'median')
        
    Returns
    -------
    float
        Threshold value
    """
    if method == 'otsu':
        from skimage.filters import threshold_otsu
        return threshold_otsu(signal)
    elif method == 'mean':
        return np.mean(signal) + np.std(signal)
    elif method == 'median':
        return np.median(signal) + 1.5 * np.std(signal)
    else:
        raise ValueError(f"Unknown method: {method}")
