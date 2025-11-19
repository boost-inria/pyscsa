"""
SCSA GPU Utilities 'under dev'
==================
GPU-accelerated versions of SCSA classes using CuPy.

Requires: cupy, cupyx
"""

import numpy as np
from typing import Optional, Tuple
import warnings

try:
    import cupy as cp
    from cupyx.scipy.sparse import diags as cp_diags
    from cupyx.scipy.linalg import eigh as cp_eigh
    from cupyx.scipy.special import gamma as cp_gamma
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from .core import SCSA1D, SCSA2D, SCSAResult, SCSABase


class GPUMixin:
    """Mixin class for GPU utilities."""
    
    @staticmethod
    def _to_gpu(array: np.ndarray) -> 'cp.ndarray':
        """Transfer array to GPU."""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available. Install with: pip install cupy-cuda11x")
        return cp.asarray(array)
    
    @staticmethod
    def _to_cpu(array: 'cp.ndarray') -> np.ndarray:
        """Transfer array from GPU to CPU."""
        return cp.asnumpy(array)


class SCSA1D_GPU(SCSA1D, GPUMixin):
    """GPU-accelerated 1D SCSA using CuPy."""
    
    def __init__(self, gmma: float = 0.5, device_id: int = 0):
        """
        Initialize GPU-accelerated 1D SCSA.
        
        Parameters
        ----------
        gmma : float, default=0.5
            Gamma parameter for SCSA computation
        device_id : int, default=0
            GPU device ID to use
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available. Install with: pip install cupy-cuda11x")
        
        super().__init__(gmma)
        cp.cuda.Device(device_id).use()
        self.device_id = device_id
    
    def _create_delta_matrix(self, n: int, fe: float = 1.0) -> 'cp.ndarray':
        """GPU version of delta matrix creation."""
        feh = 2 * cp.pi / n
        ex = cp.kron(cp.arange(n-1, 0, -1), cp.ones((n, 1)))
        
        if n % 2 == 0:
            dx = -cp.pi**2 / (3 * feh**2) - (1/6) * cp.ones((n, 1))
            test_bx = -(-1)**ex * 0.5 / (cp.sin(ex * feh * 0.5)**2)
            test_tx = -(-1)**(-ex) * 0.5 / (cp.sin((-ex) * feh * 0.5)**2)
        else:
            dx = -cp.pi**2 / (3 * feh**2) - (1/12) * cp.ones((n, 1))
            test_bx = -0.5 * ((-1)**ex) * cp.tan(ex * feh * 0.5)**-1 / cp.sin(ex * feh * 0.5)
            test_tx = -0.5 * ((-1)**(-ex)) * cp.tan((-ex) * feh * 0.5)**-1 / cp.sin((-ex) * feh * 0.5)
        
        rng = list(range(-n+1, 1)) + list(range(n-1, 0, -1))
        Ex = cp_diags(
            cp.concatenate((test_bx, dx, test_tx), axis=1).T,
            cp.array(rng),
            shape=(n, n)
        ).toarray()
        
        return (feh / fe)**2 * Ex
    
    def reconstruct(self, signal: np.ndarray, h: float = 1.0, 
                   lambda_g: Optional[float] = None) -> SCSAResult:
        """GPU-accelerated reconstruction."""
        # Transfer to GPU
        signal_gpu = self._to_gpu(signal.flatten())
        
        # Handle negative values
        min_signal = None
        if cp.min(signal_gpu) < 0:
            min_signal = float(cp.min(signal_gpu))
            signal_gpu = signal_gpu - min_signal
        
        if lambda_g is None:
            lambda_g = 0
        
        # Create delta matrix on GPU
        n = len(signal_gpu)
        D = self._create_delta_matrix(n)
        
        # SCSA computation on GPU
        Y = cp.diag(signal_gpu)
        Lcl = (1 / (2 * cp.pi**0.5)) * float(cp_gamma(self._gmma + 1) / cp_gamma(self._gmma + 1.5))
        
        # Construct Schrödinger operator
        SC = -(h**2 * D) - Y
        
        # Eigenvalue decomposition on GPU
        eigenvals, eigenvecs = cp.linalg.eigh(SC)
        
        # Select eigenvalues
        mask = eigenvals < lambda_g
        selected_eigenvals = eigenvals[mask]
        selected_eigenvecs = eigenvecs[:, mask]
        
        if len(selected_eigenvals) == 0:
            warnings.warn("No eigenvalues below threshold.")
            return SCSAResult(
                reconstructed=self._to_cpu(signal_gpu),
                eigenvalues=np.array([]),
                eigenfunctions=np.array([]),
                num_eigenvalues=0
            )
        
        # Compute kappa values
        kappa = cp.diag((lambda_g - selected_eigenvals)**self._gmma)
        
        # Normalize eigenfunctions
        norms = cp.linalg.norm(selected_eigenvecs, axis=0)
        eigenfunctions_normalized = selected_eigenvecs / norms
        
        # Reconstruct signal
        reconstructed = -lambda_g + ((h / Lcl) * 
                                     cp.sum((eigenfunctions_normalized**2) @ kappa, axis=1)
                                     )**(2 / (1 + 2*self._gmma))
        
        if min_signal is not None:
            reconstructed += min_signal
        
        # Transfer back to CPU
        reconstructed_cpu = self._to_cpu(reconstructed)
        signal_cpu = self._to_cpu(signal_gpu)
        
        # Compute metrics
        metrics = self.compute_metrics(signal_cpu, reconstructed_cpu)
        
        return SCSAResult(
            reconstructed=reconstructed_cpu,
            eigenvalues=self._to_cpu(kappa),
            eigenfunctions=self._to_cpu(eigenfunctions_normalized),
            num_eigenvalues=len(selected_eigenvals),
            metrics=metrics
        )
    
    def filter_with_c_scsa(self, signal: np.ndarray, 
                          curvature_weight: float = 4.0,
                          h_range: Optional[Tuple[float, float]] = None) -> SCSAResult:
        """GPU-accelerated C-SCSA filtering."""
        signal_gpu = self._to_gpu(signal.flatten())
        
        # Determine h range
        if h_range is None:
            h_min = float(cp.sqrt(cp.max(signal_gpu) / cp.pi))
            h_max = float(cp.max(signal_gpu) * 10)
            h_values = np.linspace(h_min, h_max, 100)
        else:
            h_values = np.linspace(h_range[0], h_range[1], 50)
        
        best_cost = float('inf')
        best_h = h_values[0]
        print(f"Optimizing h parameter on GPU {self.device_id}...")
        
        for h in h_values:
            result = self.reconstruct(self._to_cpu(signal_gpu), h)
            reconstructed_gpu = self._to_gpu(result.reconstructed)
            
            # Cost function on GPU
            accuracy_cost = float(cp.sum((signal_gpu - reconstructed_gpu)**2))
            
            grad1 = cp.gradient(reconstructed_gpu)
            grad2 = cp.gradient(grad1)
            curvature = cp.abs(grad2) / (1 + grad1**2)**1.5
            curvature_cost = float(cp.sum(curvature))
            
            mu = (10**curvature_weight) / (curvature_cost + 1e-10)
            total_cost = accuracy_cost + mu * curvature_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_h = h
        
        result = self.reconstruct(self._to_cpu(signal_gpu), best_h)
        result.optimal_h = best_h
        return result


class SCSA2D_GPU(SCSA2D, GPUMixin):
    """GPU-accelerated 2D SCSA using CuPy."""
    
    def __init__(self, gmma: float = 2.0, device_id: int = 0):
        """
        Initialize GPU-accelerated 2D SCSA.
        
        Parameters
        ----------
        gmma : float, default=2.0
            Gamma parameter for SCSA computation
        device_id : int, default=0
            GPU device ID to use
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available. Install with: pip install cupy-cuda11x")
        
        super().__init__(gmma)
        cp.cuda.Device(device_id).use()
        self.device_id = device_id
    
    def _create_diff_matrix_2d(self, M: int) -> 'cp.ndarray':
        """GPU version of 2D difference matrix."""
        delta = 2 * cp.pi / M
        delta_t = 1
        
        # Create difference indexes matrix
        k_grid, j_grid = cp.meshgrid(cp.arange(M), cp.arange(M), indexing='ij')
        diff_indexes = cp.abs(k_grid - j_grid)
        diff_indexes[diff_indexes == 0] = 1  # Temporary for computation
        
        D2_matrix = cp.ones((M, M))
        arg = diff_indexes * delta / 2
        
        if M % 2 == 0:
            D2_matrix = -(-1)**diff_indexes * (0.5 / (cp.sin(arg)**2))
            cp.fill_diagonal(D2_matrix, (-cp.pi**2 / (3 * delta**2)) - 1/6)
        else:
            D2_matrix = (-(-1)**diff_indexes) * (0.5 * (cp.cos(arg) / cp.sin(arg)) / cp.sin(arg))
            cp.fill_diagonal(D2_matrix, (-cp.pi**2 / (3 * delta**2)) - 1/12)
        
        return D2_matrix * delta**2 / delta_t**2
    
    def reconstruct(self, image: np.ndarray, h: float = 10.0,
                   lambda_g: float = 0) -> SCSAResult:
        """GPU-accelerated 2D reconstruction."""
        # Transfer to GPU
        image_gpu = self._to_gpu(image)
        
        i_length, j_length = image_gpu.shape
        
        # Handle negative values
        min_image = None
        if cp.any(image_gpu < 0):
            min_image = float(cp.min(image_gpu))
            image_gpu -= min_image
        
        # Process on GPU using batch operations where possible
        kappa = []
        rho = []
        phi_i = []
        phi_j = []
        Nh = []
        Mh = []
        
        D2 = self._create_diff_matrix_2d(max(i_length, j_length))
        
        # Process rows
        for i in range(i_length):
            y = image_gpu[i, :]
            A = -h**2 * D2[:j_length, :j_length] - cp.diag(0.5 * y)
            eigenvals, eigenvecs = cp_eigh(A)
            
            mask = eigenvals < lambda_g
            mu = eigenvals[mask]
            psi_x = eigenvecs[:, mask]
            Nx = len(mu)
            
            if Nx > 0:
                norms = cp.linalg.norm(psi_x, axis=0)
                psi_x = psi_x / norms
            
            kappa.append(mu)
            phi_i.append(psi_x)
            Nh.append(Nx)
        
        # Process columns
        for j in range(j_length):
            y = image_gpu[:, j]
            A = -h**2 * D2[:i_length, :i_length] - cp.diag(0.5 * y)
            eigenvals, eigenvecs = cp_eigh(A)
            
            mask = eigenvals < lambda_g
            mu = eigenvals[mask]
            psi_x = eigenvecs[:, mask]
            Mx = len(mu)
            
            if Mx > 0:
                norms = cp.linalg.norm(psi_x, axis=0)
                psi_x = psi_x / norms
            
            rho.append(mu)
            phi_j.append(psi_x)
            Mh.append(Mx)
        
        # Reconstruct
        L2gamma = 1 / (4 * cp.pi) * float(cp_gamma(self._gmma + 1) / cp_gamma(self._gmma + 2))
        reconstructed = cp.zeros((i_length, j_length))
        
        for i in range(i_length):
            for j in range(j_length):
                if Nh[i] > 0 and Mh[j] > 0:
                    for n in range(Nh[i]):
                        for m in range(Mh[j]):
                            reconstructed[i, j] += (
                                (lambda_g - (kappa[i][n] + rho[j][m]))**self._gmma *
                                phi_i[i][j, n]**2 * phi_j[j][i, m]**2
                            )
        
        reconstructed = -lambda_g + ((h**2 / L2gamma) * reconstructed)**(1 / (1 + self._gmma))
        
        # Transfer back to CPU
        reconstructed_cpu = self._to_cpu(reconstructed)
        image_cpu = self._to_cpu(image_gpu)
        
        metrics = self.compute_metrics(image_cpu, reconstructed_cpu)
        
        return SCSAResult(
            reconstructed=reconstructed_cpu,
            eigenvalues=[kappa, rho],
            eigenfunctions=[phi_i, phi_j],
            num_eigenvalues=int(sum(Nh) + sum(Mh)),
            metrics=metrics
        )


def benchmark_gpu_vs_cpu(signal_length: int = 1000, num_runs: int = 5):
    """
    Benchmark GPU vs CPU performance.
    
    Parameters
    ----------
    signal_length : int
        Length of test signal
    num_runs : int
        Number of benchmark runs
    """
    import time
    
    # Generate test signal
    x = np.linspace(-10, 10, signal_length)
    signal = -2 * (1/np.cosh(x))**2
    
    # CPU benchmark
    scsa_cpu = SCSA1D(gmma=0.5)
    cpu_times = []
    for _ in range(num_runs):
        start = time.time()
        scsa_cpu.reconstruct(signal, h=1.0)
        cpu_times.append(time.time() - start)
    
    cpu_avg = np.mean(cpu_times)
    
    # GPU benchmark
    if CUPY_AVAILABLE:
        scsa_gpu = SCSA1D_GPU(gmma=0.5)
        gpu_times = []
        for _ in range(num_runs):
            start = time.time()
            scsa_gpu.reconstruct(signal, h=1.0)
            cp.cuda.Stream.null.synchronize()  # Ensure GPU completion
            gpu_times.append(time.time() - start)
        
        gpu_avg = np.mean(gpu_times)
        speedup = cpu_avg / gpu_avg
        
        print(f"Signal length: {signal_length}")
        print(f"CPU average time: {cpu_avg:.4f}s")
        print(f"GPU average time: {gpu_avg:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("CuPy not available for GPU benchmarking")


if __name__ == "__main__":
    print("SCSA GPU Utilities")
    print("=" * 50)
    if CUPY_AVAILABLE:
        print(f"CuPy available: GPU acceleration enabled")
        print(f"Available GPUs: {cp.cuda.runtime.getDeviceCount()}")
    else:
        print("CuPy not available. Install with: pip install cupy-cuda11x")
