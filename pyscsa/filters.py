"""
Advanced filtering methods based on SCSA.
"""

import numpy as np
from typing import Optional, Tuple, Union
from .core import SCSA1D, SCSA2D, SCSAResult


class AdaptiveSCSA:
    """Adaptive SCSA with automatic parameter selection."""
    
    def __init__(self, base_gmma: float = 0.5):
        """
        Initialize Adaptive SCSA.
        
        Parameters
        ----------
        base_gamma : float
            Base gamma parameter
        """
        self.base_gamma = base_gmma
        self.scsa1d = SCSA1D(base_gmma)
        self.scsa2d = SCSA2D(base_gmma)
    
    def adapt_gamma(self, signal: np.ndarray) -> float:
        """
        Adaptively select gamma based on signal characteristics.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
            
        Returns
        -------
        float
            Adapted gamma value
        """
        # Estimate signal smoothness
        if signal.ndim == 1:
            grad = np.gradient(signal)
            smoothness = np.std(grad)
        else:
            grad_x, grad_y = np.gradient(signal)
            smoothness = np.mean([np.std(grad_x), np.std(grad_y)])
        
        # Map smoothness to gamma (heuristic)
        # Smoother signals -> lower gamma
        # Noisier signals -> higher gamma
        gmma = self.base_gmma * (1 + np.tanh(smoothness))
        
        return np.clip(gmma, 0.1, 5.0)
    
    def denoise(self, signal: np.ndarray, adapt_h: bool = True,
               adapt_gmma: bool = True) -> SCSAResult:
        """
        Denoise signal with adaptive parameters.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        adapt_h : bool
            Whether to adapt h parameter
        adapt_gamma : bool
            Whether to adapt gmma parameter
            
        Returns
        -------
        SCSAResult
            Denoising result
        """
        # Adapt gamma if requested
        if adapt_gmma:
            gamma = self.adapt_gmma(signal)
            if signal.ndim == 1:
                self.scsa1d.gmma = gmma
            else:
                self.scsa2d.gmma = gmma
        
        # Process based on dimensionality
        if signal.ndim == 1:
            if adapt_h:
                return self.scsa1d.filter_with_optimal_h(signal)
            else:
                return self.scsa1d.reconstruct(signal)
        else:
            if adapt_h:
                # Estimate optimal h for 2D
                h_opt = self._estimate_h_2d(signal)
                return self.scsa2d.reconstruct(signal, h=h_opt)
            else:
                return self.scsa2d.reconstruct(signal)
    
    def _estimate_h_2d(self, image: np.ndarray) -> float:
        """Estimate optimal h for 2D image."""
        # Simple heuristic based on image statistics
        noise_level = np.std(image - np.mean(image))
        h = np.sqrt(image.max() / np.pi) * (1 + noise_level)
        return np.clip(h, 1.0, 50.0)


class MultiScaleSCSA:
    """Multi-scale SCSA processing."""
    
    def __init__(self, scales: list = [1, 2, 4], gmma: float = 0.5):
        """
        Initialize Multi-scale SCSA.
        
        Parameters
        ----------
        scales : list
            List of scale factors
        gamma : float
            Gamma parameter
        """
        self.scales = scales
        self.scsa = SCSA1D(gmma) if len(scales) > 0 else SCSA2D(gmma)
    
    def decompose(self, signal: np.ndarray) -> list:
        """
        Decompose signal at multiple scales.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
            
        Returns
        -------
        list
            List of components at different scales
        """
        components = []
        
        for scale in self.scales:
            # Downsample
            if signal.ndim == 1:
                downsampled = signal[::scale]
            else:
                downsampled = signal[::scale, ::scale]
            
            # Process at this scale
            result = self.scsa.reconstruct(downsampled)
            
            # Upsample back
            if signal.ndim == 1:
                upsampled = np.interp(
                    np.arange(len(signal)),
                    np.arange(0, len(signal), scale),
                    result.reconstructed
                )
            else:
                from scipy.ndimage import zoom
                zoom_factor = (signal.shape[0] / downsampled.shape[0],
                             signal.shape[1] / downsampled.shape[1])
                upsampled = zoom(result.reconstructed, zoom_factor, order=1)
            
            components.append(upsampled)
        
        return components
    
    def reconstruct(self, signal: np.ndarray, weights: Optional[list] = None) -> np.ndarray:
        """
        Reconstruct signal from multi-scale components.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        weights : list, optional
            Weights for each scale
            
        Returns
        -------
        np.ndarray
            Reconstructed signal
        """
        components = self.decompose(signal)
        
        if weights is None:
            weights = np.ones(len(components)) / len(components)
        
        return np.sum([w * c for w, c in zip(weights, components)], axis=0)


class RobustSCSA:
    """Robust SCSA with outlier handling."""
    
    def __init__(self, gmma: float = 0.5, outlier_threshold: float = 3.0):
        """
        Initialize Robust SCSA.
        
        Parameters
        ----------
        gamma : float
            Gamma parameter
        outlier_threshold : float
            Threshold for outlier detection (in std deviations)
        """
        self.gmma = gmma
        self.outlier_threshold = outlier_threshold
        self.scsa = SCSA1D(gmma)
    
    def detect_outliers(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect outliers in signal.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
            
        Returns
        -------
        np.ndarray
            Boolean mask of outliers
        """
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        threshold = self.outlier_threshold * mad * 1.4826
        
        return np.abs(signal - median) > threshold
    
    def denoise(self, signal: np.ndarray, handle_outliers: bool = True) -> SCSAResult:
        """
        Robustly denoise signal.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        handle_outliers : bool
            Whether to handle outliers
            
        Returns
        -------
        SCSAResult
            Denoising result
        """
        if handle_outliers:
            # Detect and replace outliers
            outlier_mask = self.detect_outliers(signal)
            cleaned_signal = signal.copy()
            
            if np.any(outlier_mask):
                # Replace outliers with local median
                from scipy.ndimage import median_filter
                cleaned_signal[outlier_mask] = median_filter(signal, size=5)[outlier_mask]
        else:
            cleaned_signal = signal
        
        # Apply SCSA
        return self.scsa.filter_with_optimal_h(cleaned_signal)
