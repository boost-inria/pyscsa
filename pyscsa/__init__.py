# pyscsa/__init__.py
"""
PySCSA - Semi-Classical Signal Analysis Library
================================================

A comprehensive Python library for signal and image processing using 
Semi-Classical Signal Analysis methods.

Main Features:
    * 1D signal reconstruction and filtering (SCSA1D)
    * 2D image reconstruction (SCSA2D)
    * Automatic parameter optimization (C-SCSA)
    * Windowed processing for large images
    * Comprehensive visualization tools
"""

__version__ = "1.0.0"
__author__ = "BOOST"
__email__ = "boost@inria.fr"

from .core import (
    SCSA1D,
    SCSA2D,
    SCSABase,
    SCSAResult,
    SCSAVisualizer,
    add_noise,
    normalize_signal,
)

from .utils import (
    SignalGenerator,
    NoiseGenerator,
    compute_snr,
    compute_psnr,
)

from .filters import (
    AdaptiveSCSA,
    MultiScaleSCSA,
    RobustSCSA,
)

from .metrics import (
    QualityMetrics,
    PerformanceAnalyzer,
)

__all__ = [
    # Core classes
    'SCSA1D',
    'SCSA2D',
    'SCSABase',
    'SCSAResult',
    'SCSAVisualizer',
    # Utility functions
    'add_noise',
    'normalize_signal',
    'SignalGenerator',
    'NoiseGenerator',
    'compute_snr',
    'compute_psnr',
    # Advanced filters
    'AdaptiveSCSA',
    'MultiScaleSCSA',
    'RobustSCSA',
    # Metrics
    'QualityMetrics',
    'PerformanceAnalyzer',
]
