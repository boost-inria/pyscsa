"""
SCSA (Semi-Classical Signal Analysis) Library
==============================================
A Python library for signal and image processing using Semi-Classical Signal Analysis.

Author: boost
License: MIT
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
import warnings
from scipy.special import gamma
from scipy.integrate import simpson
from scipy.sparse import diags
from scipy.linalg import eigh
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


@dataclass
class SCSAResult:
    """Container for SCSA computation results."""
    reconstructed: np.ndarray
    eigenvalues: np.ndarray
    eigenfunctions: np.ndarray
    num_eigenvalues: int
    c_scsa: Optional[float] = None
    metrics: Optional[dict] = None


class SCSABase:
    """Base class for SCSA implementations."""
    
    def __init__(self, gmma: float = 0.5):
        """
        Initialize SCSA base class.
        
        Parameters
        ----------
        gmma : float, default=0.5
            Gamma parameter for SCSA computation.
        """
        self._gmma = gmma
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate input parameters."""
        if self._gmma <= 0:
            raise ValueError("Gamma must be positive")
    
    @staticmethod
    def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
        """
        Compute quality metrics between original and reconstructed signals.
        
        Parameters
        ----------
        original : np.ndarray
            Original signal
        reconstructed : np.ndarray
            Reconstructed signal
            
        Returns
        -------
        dict
            Dictionary containing MSE, RMSE, PSNR, and SNR metrics
        """
        mse = mean_squared_error(original.flatten(), reconstructed.flatten())
        rmse = np.sqrt(mse)
        
        # PSNR calculation
        max_val = np.max(original)
        psnr = 20 * np.log10(max_val / rmse) if rmse > 0 else float('inf')
        
        # SNR calculation
        signal_power = np.mean(original**2)
        noise_power = np.mean((original - reconstructed)**2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        return {
            'mse': mse,
            'rmse': rmse,
            'psnr': psnr,
            'snr': snr
        }


class SCSA1D(SCSABase):
    """
    1D Semi-Classical Signal Analysis for signal reconstruction and filtering.
    
    This class implements both standard SCSA and C-SCSA (with automatic h optimization)
    for 1D signal processing.
    """
    
    def __init__(self, gmma: float = 0.5):
        """
        Initialize 1D SCSA.
        
        Parameters
        ----------
        gmma : float, default=0.5
            Gamma parameter for SCSA computation.
        """
        super().__init__(gmma)
    
    def _create_delta_matrix(self, n: int, fe: float = 1.0) -> np.ndarray:
        """
        Create the discretization matrix D2 for the differential operator.
        
        Parameters
        ----------
        n : int
            Size of the matrix
        fe : float, default=1.0
            Sampling parameter
            
        Returns
        -------
        np.ndarray
            Delta matrix for discretization
        """
        feh = 2 * np.pi / n
        ex = np.kron(np.arange(n-1, 0, -1), np.ones((n, 1)))
        
        if n % 2 == 0:
            dx = -np.pi**2 / (3 * feh**2) - (1/6) * np.ones((n, 1))
            test_bx = -(-1)**ex * 0.5 / (np.sin(ex * feh * 0.5)**2)
            test_tx = -(-1)**(-ex) * 0.5 / (np.sin((-ex) * feh * 0.5)**2)
        else:
            dx = -np.pi**2 / (3 * feh**2) - (1/12) * np.ones((n, 1))
            test_bx = -0.5 * ((-1)**ex) * np.tan(ex * feh * 0.5)**-1 / np.sin(ex * feh * 0.5)
            test_tx = -0.5 * ((-1)**(-ex)) * np.tan((-ex) * feh * 0.5)**-1 / np.sin((-ex) * feh * 0.5)
        
        rng = list(range(-n+1, 1)) + list(range(n-1, 0, -1))
        Ex = diags(
            np.concatenate((test_bx, dx, test_tx), axis=1).T,
            np.array(rng),
            shape=(n, n)
        ).toarray()
        
        return (feh / fe)**2 * Ex
    
    def reconstruct(self, signal: np.ndarray, h: float = 1.0, 
                   lambda_g: Optional[float] = None) -> SCSAResult:
        """
        Reconstruct a 1D signal using SCSA.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal (will be converted to positive if negative)
        h : float, default=1.0
            Semi-classical parameter
        lambda_g : float, optional
            Lambda parameter. If None, set to 0
            
        Returns
        -------
        SCSAResult
            Object containing the following attributes:
            - reconstructed: Reconstructed signal
            - eigenvalues: Eigenvalues used in reconstruction
            - eigenfunctions: Eigenfunctions used in reconstruction
            - num_eigenvalues: Number of eigenvalues used
            - metrics: Quality metrics dictionary
        """
        # Ensure signal is positive for SCSA
        min_signal = None
        if signal.min() < 0:
            min_signal = signal.min()
            signal = signal - min_signal
        signal = signal.flatten()
        # print("Signal with min value adjusted:", signal.min())
        # print("Signal min value:", min_signal)

        if lambda_g is None:
            lambda_g = 0
        
        # Create delta matrix
        n = len(signal)
        D = self._create_delta_matrix(n)
        
        # SCSA computation
        Y = np.diag(signal)
        Lcl = (1 / (2 * np.pi**0.5)) * (gamma(self._gmma + 1) / gamma(self._gmma + 1.5))
        
        # Construct Schrödinger operator
        SC = -(h**2 * D) - Y
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(SC)
        
        # Select eigenvalues below threshold
        mask = eigenvals < lambda_g
        #print(f"number of eigenvalues lower than zero: {np.sum(mask)}")
        selected_eigenvals = eigenvals[mask]
        selected_eigenvecs = eigenvecs[:, mask]
        
        if len(selected_eigenvals) == 0:
            warnings.warn("No eigenvalues below threshold. Returning original signal.")
            return SCSAResult(
                reconstructed=signal,
                eigenvalues=np.array([]),
                eigenfunctions=np.array([]),
                num_eigenvalues=0
            )
        
        # Compute kappa values
        kappa = np.diag((lambda_g - selected_eigenvals)**self._gmma)
        
        # Normalize eigenfunctions
        eigenfunctions_normalized = normalize(selected_eigenvecs, norm = 'l2', axis = 0)
        
        # Reconstruct signal
        reconstructed = -lambda_g + ((h / Lcl) * 
                                     np.sum((eigenfunctions_normalized**2) @ kappa, axis=1)
                                     )**(2 / (1 + 2*self._gmma))
        if  min_signal is not None:
            reconstructed += min_signal
        # Compute metrics
        metrics = self.compute_metrics(signal, reconstructed)
        
        return SCSAResult(
            reconstructed=reconstructed,
            eigenvalues=kappa,
            eigenfunctions=eigenfunctions_normalized,
            num_eigenvalues=len(selected_eigenvals),
            metrics=metrics
        )
    
    def filter_with_c_scsa(self, signal: np.ndarray, 
                            curvature_weight: float = 4.0,
                            h_range: Optional[Tuple[float, float]] = None) -> SCSAResult:
        """
        Filter 1D signal using C-SCSA with automatic h optimization.
        
        Parameters
        ----------
        signal : np.ndarray
            Input noisy signal
        curvature_weight : float, default=4.0
            Weight for curvature penalty in cost function
        h_range : Tuple[float, float], optional
            Range for h parameter search. If None, automatically determined
            
        Returns
        -------
        SCSAResult
            Object containing filtered signal and optimal parameters
        """
                                
        # Determine h range
        if h_range is None:
            h_min = np.sqrt(signal.max() / np.pi)
            h_max = signal.max() * 10
            h_values = np.linspace(h_min, h_max, 100)
        else:
            h_values = np.linspace(h_range[0], h_range[1], 50)
        
        best_cost = float('inf')
        best_h = h_values[0]
        print("Optimizing h parameter in the range:", h_values[0], "to", h_values[-1])
        #original signal for fair comparison in the cost function
        # Grid search for optimal h
        for h in h_values:
            result = self.reconstruct(signal, h)
            reconstructed = result.reconstructed
            
            # Cost function components
            # Accuracy penalty
            accuracy_cost = np.sum((signal - reconstructed)**2)
            
            # Curvature penalty
            grad1 = np.gradient(reconstructed)
            grad2 = np.gradient(grad1)
            curvature = np.abs(grad2) / (1 + grad1**2)**1.5
            curvature_cost = np.sum(curvature)
            
            # Total cost with regularization
            mu = (10**curvature_weight) / (np.sum(curvature) + 1e-10)
            total_cost = accuracy_cost + mu * curvature_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_h = h
                best_result = result
        

        result = self.reconstruct(signal, best_h)
        result.optimal_h = best_h
        return result

    def denoise(self, noisy_signal: np.ndarray, **kwargs) -> np.ndarray:
        """
        Convenience method for signal denoising.
        
        Parameters
        ----------
        noisy_signal : np.ndarray
            Input noisy signal
        **kwargs
            Additional parameters passed to filter_with_c_scsa
            
        Returns
        -------
        np.ndarray
            Denoised signal
        """
        result = self.filter_with_c_scsa(noisy_signal, **kwargs)
        return result.reconstructed


class SCSA2D(SCSABase):
    """
    2D Semi-Classical Signal Analysis for image reconstruction.
    
    This class implements 2D SCSA using separation of variables approach
    for image processing and reconstruction.
    """
    
    def __init__(self, gmma: float = 2.0):
        """
        Initialize 2D SCSA.
        
        Parameters
        ----------
        gmma : float, default=2.0
            Gamma parameter for SCSA computation.
        """
        super().__init__(gmma)
    
    def _create_diff_matrix_2d(self, M: int) -> np.ndarray:
        """
        Create difference matrix for 2D SCSA.
        
        Parameters
        ----------
        M : int
            Matrix dimension
            
        Returns
        -------
        np.ndarray
            2D difference matrix
        """
        delta = 2 * np.pi / M
        delta_t = 1
        
        # Create difference indexes matrix
        diff_indexes = np.ones((M, M), dtype=np.int64)
        for k in range(M):
            for j in range(M):
                if k != j:
                    diff_indexes[k, j] = abs(k - j)
        
        D2_matrix = np.ones((M, M))
        arg = diff_indexes * delta / 2
        
        if M % 2 == 0:  # Even M
            D2_matrix = -(-1)**diff_indexes * (0.5 / (np.sin(arg)**2))
            D2_matrix[np.eye(M, dtype=bool)] = (-np.pi**2 / (3 * delta**2)) - 1/6
        else:  # Odd M
            D2_matrix = (-(-1)**diff_indexes) * (0.5 * (np.cot(arg) / np.sin(arg)))
            D2_matrix[np.eye(M, dtype=bool)] = (-np.pi**2 / (3 * delta**2)) - 1/12
        
        return D2_matrix * delta**2 / delta_t**2
    
    def _scsa_1d_for_2d(self, y: np.ndarray, h: float, 
                       lam: float) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        1D SCSA helper for 2D separation of variables.
        
        Parameters
        ----------
        y : np.ndarray
            1D signal slice
        h : float
            Semi-classical parameter
        lam : float
            Lambda threshold
            
        Returns
        -------
        Tuple containing eigenvalues, eigenvectors, and count
        """
        M = len(y)
        D2 = self._create_diff_matrix_2d(M)
        A = -h**2 * D2 - np.diag(0.5 * y)
        
        eigenvals, eigenvecs = eigh(A)
        
        mask = eigenvals < lam
        mu = eigenvals[mask]
        psi_x = eigenvecs[:, mask]
        Nx = len(mu)
        
        if Nx > 0:
            psi_x = np.apply_along_axis(lambda v: v / np.linalg.norm(v), 0, psi_x)
        
        return mu, psi_x, Nx
    
    def reconstruct(self, image: np.ndarray, h: float = 10.0,
                   lambda_g: float = 0) -> SCSAResult:
        """
        Reconstruct a 2D image using SCSA with separation of variables.
        
        Parameters
        ----------
        image : np.ndarray
            Input 2D image
        h : float, default=10.0
            Semi-classical parameter
        lambda_g : float, default=0
            Lambda threshold parameter
            
        Returns
        -------
        SCSAResult
            Object containing the following attributes:
            - reconstructed: Reconstructed image
            - eigenvalues: Eigenvalues from rows and columns
            - eigenfunctions: Eigenfunctions from rows and columns
            - num_eigenvalues: Total number of eigenvalues used
            - metrics: Quality metrics dictionary
        """
        i_length, j_length = image.shape
        
        # Ensure image is positive for SCSA
        min_image = None
        if np.any(image < 0):
            min_image = image.min()
            image -= min_image
        image = image.astype(float)

        # Initialize storage
        kappa = [None] * i_length
        rho = [None] * j_length
        phi_i = [None] * i_length
        phi_j = [None] * j_length
        Nh = np.zeros(i_length, dtype=int)
        Mh = np.zeros(j_length, dtype=int)
        
        # Compute 1D SCSA for each row and column
        for i in range(i_length):
            kappa[i], phi_i[i], Nh[i] = self._scsa_1d_for_2d(image[i, :], h, lambda_g)
        
        for j in range(j_length):
            rho[j], phi_j[j], Mh[j] = self._scsa_1d_for_2d(image[:, j], h, lambda_g)
        
        # Reconstruct image
        L2gamma = 1 / (4 * np.pi) * gamma(self._gmma + 1) / gamma(self._gmma + 2)
        reconstructed = np.zeros((i_length, j_length))
        
        for i in range(i_length):
            for j in range(j_length):
                for n in range(Nh[i]):
                    for m in range(Mh[j]):
                        reconstructed[i, j] += (
                            (lambda_g - (kappa[i][n] + rho[j][m]))**self._gmma *
                            phi_i[i][j, n]**2 * phi_j[j][i, m]**2
                        )
        
        reconstructed = -lambda_g + ((h**2 / L2gamma) * reconstructed)**(1 / (1 + self._gmma))
        
        # Compute metrics
        metrics = self.compute_metrics(image, reconstructed)
        
        return SCSAResult(
            reconstructed=reconstructed,
            eigenvalues=[kappa, rho],
            eigenfunctions=[phi_i, phi_j],
            num_eigenvalues=int(np.sum(Nh) + np.sum(Mh)),
            metrics=metrics
        )
    
    def reconstruct_windowed(self, image: np.ndarray, h: float = 10.0,
                           window_size: int = 4, stride: int = 1,
                           lambda_g: float = 0) -> np.ndarray:
        """
        Reconstruct image using windowed SCSA approach.
        
        Parameters
        ----------
        image : np.ndarray
            Input 2D image
        h : float, default=10.0
            Semi-classical parameter
        window_size : int, default=4
            Size of sliding window
        stride : int, default=1
            Stride for sliding window
        lambda_g : float, default=0
            Lambda threshold
            
        Returns
        -------
        np.ndarray
            Reconstructed image
        """
        rows, cols = image.shape
        result = np.zeros_like(image, dtype=float)
        weight_map = np.zeros_like(image, dtype=float)
        
        for i in range(0, rows - window_size + 1, stride):
            for j in range(0, cols - window_size + 1, stride):
                window = image[i:i + window_size, j:j + window_size]
                
                # Process window
                window_result = self.reconstruct(window, h, lambda_g)
                
                # Accumulate results with overlapping
                result[i:i + window_size, j:j + window_size] += window_result.reconstructed
                weight_map[i:i + window_size, j:j + window_size] += 1
        
        # Average overlapping regions
        result = np.divide(result, weight_map, where=weight_map > 0)
        
        return result
    
    def denoise(self, noisy_image: np.ndarray, method: str = 'windowed',
               **kwargs) -> np.ndarray:
        """
        Convenience method for image denoising.
        
        Parameters
        ----------
        noisy_image : np.ndarray
            Input noisy image
        method : str, default='windowed'
            Method to use ('standard' or 'windowed')
        **kwargs
            Additional parameters passed to reconstruction method
            
        Returns
        -------
        np.ndarray
            Denoised image
        """
        if method == 'windowed':
            return self.reconstruct_windowed(noisy_image, **kwargs)
        else:
            result = self.reconstruct(noisy_image, **kwargs)
            return result.reconstructed


class SCSAVisualizer:
    """Visualization utilities for SCSA results."""
    
    @staticmethod
    def plot_1d_comparison(original: np.ndarray, reconstructed: np.ndarray,
                          title: str = "SCSA 1D Reconstruction",
                          figsize: Tuple[int, int] = (12, 5)):
        """
        Plot comparison between original and reconstructed 1D signals.
        
        Parameters
        ----------
        original : np.ndarray
            Original signal
        reconstructed : np.ndarray
            Reconstructed signal
        title : str
            Plot title
        figsize : Tuple[int, int]
            Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Signal comparison
        axes[0].plot(original, 'k-', label='Original', linewidth=2)
        axes[0].plot(reconstructed, 'b--', label='SCSA', linewidth=1.5, alpha=0.8)
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("Value")
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error plot
        error = np.abs(original - reconstructed)
        relative_error = error / (np.abs(original) + 1e-10) * 100
        
        axes[1].plot(relative_error, 'r-', linewidth=1.5)
        axes[1].set_xlabel("Index")
        axes[1].set_ylabel("Relative Error (%)")
        axes[1].set_title("Reconstruction Error")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_2d_comparison(original: np.ndarray, reconstructed: np.ndarray,
                          title: str = "SCSA 2D Reconstruction",
                          figsize: Tuple[int, int] = (15, 5),
                          cmap: str = 'gray'):
        """
        Plot comparison between original and reconstructed 2D images.
        
        Parameters
        ----------
        original : np.ndarray
            Original image
        reconstructed : np.ndarray
            Reconstructed image
        title : str
            Plot title
        figsize : Tuple[int, int]
            Figure size
        cmap : str
            Colormap for display
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        im1 = axes[0].imshow(original, cmap=cmap)
        axes[0].set_title("Original")
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # Reconstructed image
        im2 = axes[1].imshow(reconstructed, cmap=cmap)
        axes[1].set_title("SCSA Reconstructed")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # Difference image
        diff = np.abs(original - reconstructed)
        im3 = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title("Absolute Difference")
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_metrics(metrics: dict, title: str = "SCSA Performance Metrics"):
        """
        Create a bar plot of performance metrics.
        
        Parameters
        ----------
        metrics : dict
            Dictionary of metrics
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        bars = ax.bar(labels, values)
        
        # Color code bars
        colors = ['green' if v < 1 else 'orange' if v < 10 else 'red' 
                 for v in values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.set_yscale('log')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig


# Utility functions
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


# Example usage functions
def example_1d_reconstruction():
    """Example of 1D signal reconstruction using SCSA."""
    # Generate test signal
    x = np.linspace(-10, 10, 500)
    signal = -2 * (1/np.cosh(x))**2
    
    # Add noise
    noisy_signal = add_noise(signal, snr_db=20, seed=42)
    
    # Create SCSA instance
    scsa = SCSA1D(gmma=0.5)
    
    # Reconstruct with optimal h
    result = scsa.filter_with_c_scsa(noisy_signal)
    
    print(f"Optimal h: {result.optimal_h:.2f}")
    print(f"Number of eigenvalues: {result.num_eigenvalues}")
    print(f"Metrics: {result.metrics}")
    
    # Visualize
    viz = SCSAVisualizer()
    fig = viz.plot_1d_comparison(noisy_signal, result.reconstructed)
    plt.show()
    
    return result


def example_2d_reconstruction():
    """Example of 2D image reconstruction using SCSA."""
    # Generate test image (e.g., Gaussian blob)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    image = np.exp(-(X**2 + Y**2) / 2)
    
    # Add noise
    noisy_image = add_noise(image, snr_db=15, seed=42)
    
    # Create SCSA instance
    scsa = SCSA2D(gmma=2.0)
    
    # Reconstruct
    denoised = scsa.denoise(noisy_image, method='windowed', 
                           window_size=8, h=5.0)
    
    # Visualize
    viz = SCSAVisualizer()
    fig = viz.plot_2d_comparison(noisy_image, denoised)
    plt.show()
    
    return denoised


if __name__ == "__main__":
    print("SCSA Library - Semi-Classical Signal Analysis")
    print("=" * 50)
    print("Available classes:")
    print("  - SCSA1D: 1D signal reconstruction and filtering")
    print("  - SCSA2D: 2D image reconstruction")
    print("  - SCSAVisualizer: Visualization utilities")
    print("\nRun example_1d_reconstruction() or example_2d_reconstruction() to see demos.")
