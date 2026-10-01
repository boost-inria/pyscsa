"""
SCSA (Semi-Classical Signal Analysis) Library
==============================================
A Python library for signal and image processing using Semi-Classical Signal Analysis.

Author: boost
License: AGPL-3.0
Version: 1.0.0
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import warnings
from scipy.special import gamma
from scipy.integrate import simpson
from scipy.sparse import diags
from scipy.linalg import eigh
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


@dataclass
class SCSAResult:
    """Container for SCSA computation results."""
    reconstructed: np.ndarray
    eigenvalues: np.ndarray
    kappas: np.ndarray
    eigenfunctions: np.ndarray
    num_eigenvalues: int
    c_scsa: Optional[float] = None
    metrics: Optional[dict] = None
    # C-SCSA diagnostics (set by SCSA1D.filter_with_c_scsa)
    optimal_h: Optional[float] = None
    h_values: Optional[np.ndarray] = None
    costs: Optional[np.ndarray] = None
    sigma_hat: Optional[float] = None


def simp_integral(y: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute numerical integral using composite Simpson's rule.
    
    Custom implementation that operates on a 2D array where each column
    is integrated independently (suited for squared eigenfunctions).
    
    Parameters
    ----------
    y : np.ndarray
        2D array of shape (n, k) where each column is a discrete function
        to integrate. Each y_i represents a discrete sample at time i*dt.
    dt : float
        Sampling interval (spacing between discrete points).
        
    Returns
    -------
    np.ndarray
        1D array of shape (k,) with the integral value for each column.
    """
    n = y.shape[0]
    if n > 1:
        I = (1 / 3) * (y[0, :] + y[1, :]) * dt
        for i in range(2, n):
            if i % 2 == 0:
                I += (1 / 3) * (y[i - 1, :] + y[i, :]) * dt
            else:
                I += (y[i - 1, :] + (1 / 3) * y[i, :]) * dt
    else:
        I = y[0, :] * dt
    return I


def _curvature(v: np.ndarray) -> float:
    """Total absolute curvature of a 1-D curve, |y''| / (1 + y'^2)^(3/2)."""
    g1 = np.gradient(v)
    g2 = np.gradient(g1)
    return float(np.sum(np.abs(g2) / (1.0 + g1 ** 2) ** 1.5))


class SCSABase:
    """Base class for SCSA implementations."""
    
    def __init__(self, gmma: float = 0.5,fe: float = 1.0):
        """
        Initialize SCSA base class.
        
        Parameters
        ----------
        gmma : float, default=0.5
            Gamma parameter for SCSA computation.
        fe : float, default=1.0
            Sampling parameter for discretization.
        """
        self._gmma = gmma
        self.fe = fe
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate input parameters."""
        if self._gmma <= 0:
            raise ValueError("Gamma must be positive")

    def normalize_eigenfunctions(self, eigenvecs: np.ndarray, dx: float = 1.0,
                                 method: str = 'scipy') -> np.ndarray:
        """
        Normalize eigenfunctions so that integral(|psi|^2, dx) = 1 for each
        eigenfunction, using Simpson's rule for numerical integration.
        
        Parameters
        ----------
        eigenvecs : np.ndarray
            Matrix of eigenvectors, shape (n, k), where each column is an
            eigenfunction.
        dx : float, default=1.0
            Sampling interval (spacing between discrete points).
        method : str, default='scipy'
            Integration method: 'scipy' uses scipy.integrate.simpson,
            'custom' uses the manual composite Simpson implementation.
            
        Returns
        -------
        np.ndarray
            Normalized eigenfunctions with the same shape as input.
        """        
        if method == 'scipy':
            # scipy.integrate.simpson along axis=0 (rows) for each column
            psi_sq = eigenvecs**2
            norms_sq = simpson(psi_sq, dx=dx, axis=0)
            assert len(norms_sq) == eigenvecs.shape[1], "Expected norms shape to match number of eigenfunctions"
        elif method == 'custom':
            psi_sq = eigenvecs**2
            norms_sq = simp_integral(psi_sq, dx)
            assert len(norms_sq) == eigenvecs.shape[1], "Expected norms shape to match number of eigenfunctions"
        elif method == 'trapezoidal':
            psi = np.copy(eigenvecs)
            for i in range(psi.shape[1]):
                norms_sq = np.sqrt(np.trapezoid(psi[:, i]**2, dx=dx))
                psi[:, i] /= norms_sq
            return psi
        else:
            raise ValueError(f"Unknown integration method: {method}. "
                             "Use 'scipy' or 'custom'.")
        
        norms = np.sqrt(np.abs(norms_sq))
        # Avoid division by zero
        norms = np.where(norms < 1e-15, 1.0, norms)
        
        return eigenvecs / norms[np.newaxis, :]
    
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
    
    def __init__(self, gmma: float = 0.5, fe: float = 1.0):
        """
        Initialize 1D SCSA.
        
        Parameters
        ----------
        gmma : float, default=0.5
            Gamma parameter for SCSA computation.
        fe : float, default=1.0
            Sampling parameter for discretization.
        """
        super().__init__(gmma, fe)


    
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
                   lambda_g: Optional[float] = None, method_norm: str = 'trapezoidal') -> SCSAResult:
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
        D = self._create_delta_matrix(n, fe=self.fe)
        
        # SCSA computation
        Y = np.diag(signal)
        Lcl = (1 / (2 * np.pi**0.5)) * (gamma(self._gmma + 1) / gamma(self._gmma + 1.5))
        
        # Construct Schrödinger operator
        SC = -(h**2)* D - Y
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(SC)
        
        # Select eigenvalues below threshold
        mask = eigenvals < lambda_g
        #print(f"number of eigenvalues lower than zero: {np.sum(mask)}")
        selected_eigenvals = eigenvals[mask]
        selected_eigenvecs = eigenvecs[:, mask]
        
        if len(selected_eigenvals) == 0:
            warnings.warn("No eigenvalues below threshold. Returning original signal.")
            original = signal + min_signal if min_signal is not None else signal
            return SCSAResult(
                reconstructed=original,
                eigenvalues=eigenvals,
                kappas=np.array([]),
                eigenfunctions=np.array([]),
                num_eigenvalues=0
            )
        
        # Compute kappa values
        kappas = np.diag((lambda_g - selected_eigenvals)**self._gmma)
        
        # Normalize eigenfunctions using Simpson's rule integration
        eigenfunctions_normalized = self.normalize_eigenfunctions(selected_eigenvecs,
                                                                  dx=self.fe, method=method_norm
        )
        
        # Reconstruct signal
        reconstructed = -lambda_g + ((h / Lcl) * 
                                     np.sum((eigenfunctions_normalized**2) @ kappas, axis=1)
                                     )**(2 / (1 + 2*self._gmma))
        if  min_signal is not None:
            reconstructed += min_signal
        # Compute metrics
        metrics = self.compute_metrics(signal, reconstructed)
        
        return SCSAResult(
            reconstructed=reconstructed,
            eigenvalues=eigenvals,
            kappas=kappas,
            eigenfunctions=eigenfunctions_normalized,
            num_eigenvalues=len(selected_eigenvals),
            metrics=metrics
        )
    
    def filter_with_c_scsa(self, signal: np.ndarray,
                           curvature_weight: float = 1.5,
                           h_range: Optional[Tuple[float, float]] = None,
                           n_h: int = 40,
                           cost_fn: Optional[Callable[[np.ndarray, np.ndarray, float], float]] = None
                           ) -> SCSAResult:
        """
        Filter a 1-D signal with C-SCSA: choose h automatically from the noisy
        signal alone, with no knowledge of the clean signal or the noise level.

        The cost balances fidelity to the measurement against the roughness of the
        reconstruction. The search runs on a copy of the signal scaled to unit
        peak-to-peak, and both terms are normalised by references computed
        **once** from that copy, which makes them dimensionless and O(1):

            y_n       = y / ptp(y)                  amplitude normalisation
            sigma_hat = std(diff(y_n)) / sqrt(2)    high-pass noise estimate
            A_ref     = n * sigma_hat**2            residual expected at the noise floor
            C_ref     = curvature(y_n)              curvature of the noisy signal

            cost(h) = ||y_n - y_h||^2 / A_ref  +  w * curvature(y_h) / C_ref

        The first term falls towards 1 as the fit approaches the noise floor and
        keeps falling below 1 once the reconstruction starts absorbing noise; the
        second penalises the roughness that absorbing noise produces. Their minimum
        is the operating point.

        SCSA is exactly equivariant under ``y -> k*y, h -> sqrt(k)*h``, so the
        amplitude normalisation is lossless: the reconstruction, ``optimal_h``,
        ``h_values`` and ``sigma_hat`` are all returned in the caller's units, and
        the selected h does not depend on the signal's amplitude or units. (Without
        it, the ``(1 + y'^2)^(3/2)`` factor in the curvature makes the effective
        weight grow with amplitude, so the same signal in mmHg and in normalised
        units would be filtered differently.)

        The selection does still depend on **sampling density**. With finer
        sampling the per-sample second difference of a smooth reconstruction
        shrinks while that of the noise does not, which lowers the effective weight
        and biases the choice towards smaller h (under-smoothing). Calibrate w at
        the sampling rate you intend to use.

        .. note:: **Breaking change from versions <= 1.0.0.** The previous
           implementation set ``mu = 10**curvature_weight / sum(curvature)`` and
           multiplied it by ``sum(curvature)``, so the penalty collapsed to the
           constant ``10**curvature_weight`` and the search always returned the
           smallest h in the range. ``curvature_weight`` is now the weight ``w``
           itself, so values carried over from the old API (e.g. 4.0) are not
           comparable to the new default. The search also now runs on the
           amplitude-normalised signal described above.

        Parameters
        ----------
        signal : np.ndarray
            Noisy input signal.
        curvature_weight : float, default=1.5
            Weight w on the roughness term. The default was calibrated over three
            signal families (single pulse, arterial pressure, EEG burst) at five
            SNR levels from 6 to 20 dB, and lands within 1.5x of the best
            achievable error on 93% of those cases.

            w is roughly as sensitive as h itself (choosing it badly costs a median
            2.4x the best achievable error), so it is not a soft knob, and it does
            not transfer reliably across signal types: on synthetic test signals
            the best w ranged from about 2 (oscillatory bursts) to about 10 (a
            pulse on a long flat baseline), and w = 1.5 was within 1.5x of the best
            error in only about two thirds of cases. **Calibrate it once on a
            representative corpus of the signals you will actually process, at the
            sampling rate you will use, and then hold it fixed.** Re-tuning w per
            signal against a known clean reference is oracle tuning, and results
            obtained that way must not be reported as automatic selection.

            Below about w = 0.75 the cost has no interior minimum: the fine end of
            the grid always wins and the edge-of-range warning fires. A moderately
            wrong w, by contrast, still produces an interior minimum and no
            warning, while costing up to about 2x the best achievable error.
        h_range : tuple of (float, float), optional
            Search range for h, in the same units as the ``h`` of ``reconstruct``
            applied to the unscaled signal. If None, derived from the Weyl law
            N_h ~ (1/pi*h) * integral(sqrt(y)), targeting roughly 2 components at
            the coarse end and n/4 at the fine end.
        n_h : int, default=40
            Number of h values in the (logarithmically spaced) search grid.
        cost_fn : callable, optional
            Custom criterion ``cost_fn(signal, reconstructed, h) -> float`` to
            minimise instead of the default, called with the original signal and
            with the reconstruction and h in the caller's units. Use it to
            substitute a discrepancy principle, GCV, or an L-curve criterion
            without forking the method.

        Returns
        -------
        SCSAResult
            The reconstruction at the selected h, with diagnostics attached:

            ``optimal_h``   selected h, in the caller's units
            ``h_values``    the search grid, in the caller's units
            ``costs``       cost at each grid point (dimensionless)
            ``sigma_hat``   estimated noise standard deviation, in signal units

            Always plot ``costs`` against ``h_values`` before trusting the result.
            A flat or monotone curve means the criterion did not identify a
            minimum for this signal, and the returned h is then not meaningful.
        """
        y = np.asarray(signal, dtype=float).flatten()
        n = y.size
        if n < 8:
            raise ValueError("signal too short for C-SCSA")

        # --- amplitude normalisation ------------------------------------------
        # SCSA maps y -> k*y, h -> sqrt(k)*h exactly onto a reconstruction scaled
        # by k, so the search runs on a unit peak-to-peak copy and the results are
        # mapped back afterwards.
        scale = float(np.ptp(y))
        if scale <= 0:
            raise ValueError("signal is constant; nothing to decompose")
        yn = y / scale
        h_scale = np.sqrt(scale)

        # --- search range (normalised units) ----------------------------------
        if h_range is None:
            integral_sqrt = float(np.trapezoid(np.sqrt(np.maximum(yn - yn.min(), 0.0)),
                                               dx=self.fe))
            h_hi = integral_sqrt / (2.0 * np.pi)          # ~2 components
            h_lo = 4.0 * integral_sqrt / (np.pi * n)      # ~n/4 components
            if h_lo >= h_hi:
                h_lo, h_hi = 0.5 * h_hi, 2.0 * h_hi
        else:
            h_lo, h_hi = float(h_range[0]), float(h_range[1])
            if not (0 < h_lo < h_hi):
                raise ValueError("h_range must satisfy 0 < h_min < h_max")
            h_lo, h_hi = h_lo / h_scale, h_hi / h_scale

        h_grid = np.exp(np.linspace(np.log(h_lo), np.log(h_hi), int(n_h)))
        h_values = h_grid * h_scale                       # caller's units

        # --- fixed normalisers, computed once from the normalised signal ------
        sigma_hat = float(np.std(np.diff(yn)) / np.sqrt(2.0))
        a_ref = max(n * sigma_hat ** 2, 1e-12)
        c_ref = max(_curvature(yn), 1e-12)

        # --- search -----------------------------------------------------------
        costs = np.empty(h_grid.size)
        for i, h in enumerate(h_grid):
            rec = self.reconstruct(yn.copy(), h=float(h)).reconstructed
            if cost_fn is None:
                costs[i] = (np.sum((yn - rec) ** 2) / a_ref
                            + curvature_weight * _curvature(rec) / c_ref)
            else:
                costs[i] = float(cost_fn(y, rec * scale, float(h_values[i])))

        best = int(np.argmin(costs))
        if best in (0, h_grid.size - 1):
            warnings.warn(
                "C-SCSA selected h at the edge of the search range "
                f"({h_values[best]:.4g}); the minimum is not bracketed. Widen "
                "h_range, or inspect the returned costs before using this result.",
                RuntimeWarning)

        # Final reconstruction on the original signal, so eigenvalues, kappas and
        # metrics are in the caller's units, exactly as from reconstruct().
        result = self.reconstruct(y.copy(), h=float(h_values[best]))
        result.optimal_h = float(h_values[best])
        result.h_values = h_values
        result.costs = costs
        result.sigma_hat = sigma_hat * scale
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
            psi_x = self.normalize_eigenfunctions(psi_x, dx=1.0, method='scipy')
        
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
            kappas=np.array([k for k_list in kappa if k_list is not None for k in k_list]),
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
