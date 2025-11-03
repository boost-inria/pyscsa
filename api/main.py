from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from pyscsa import SCSA1D
from functools import lru_cache
from typing import Literal

app = FastAPI(title="SCSA API", version="1.0.0")

# More restrictive CORS in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Replace with specific origins in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    max_age=3600,
)

class SCSARequest(BaseModel):
    gamma: float = Field(default=0.5, gt=0, le=1, description="SCSA gamma parameter")
    h: float = Field(default=1.0, gt=0, description="Reconstruction parameter h")
    noise: float = Field(default=0.1, ge=0, description="Noise level")
    signal_type: Literal["gaussian", "sech", "double_well", "chirp"] = Field(
        default="gaussian", description="Type of signal to generate"
    )
    method: Literal["reconstruct", "denoise", "filter_optimal"] = Field(
        default="reconstruct", description="SCSA processing method"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "gamma": 0.5,
                "h": 1.0,
                "noise": 0.1,
                "signal_type": "gaussian",
                "method": "reconstruct"
            }
        }


def ensure_positivity(signal: np.ndarray) -> np.ndarray:
    """Ensure signal is positive for SCSA by shifting if necessary."""
    if signal.min() < 0:
        signal = signal - signal.min()
    return signal.flatten()


@lru_cache(maxsize=32)
def get_x_values(num_points: int = 200) -> np.ndarray:
    """Cache x values to avoid repeated creation."""
    return np.linspace(-10, 10, num_points)


def generate_signal(signal_type: str, x: np.ndarray) -> np.ndarray:
    """Generate various test signals."""
    signal_map = {
        "gaussian": lambda x: np.exp(-x**2),
        "sech": lambda x: 1 / np.cosh(x),
        "double_well": lambda x: -50 * (1/np.cosh(x-3))**2 - 50 * (1/np.cosh(x+3))**2,
        "chirp": lambda x: np.sin(x**2),
    }
    
    signal_func = signal_map.get(signal_type, signal_map["gaussian"])
    return ensure_positivity(signal_func(x))


@app.post("/api/scsa")
async def run_scsa(req: SCSARequest):
    """Process signal using SCSA method."""
    try:
        # Generate signal
        x = get_x_values()
        signal = generate_signal(req.signal_type, x)
        
        # Add noise with fixed seed for reproducibility (optional)
        rng = np.random.default_rng()
        noisy = ensure_positivity(signal + req.noise * rng.standard_normal(len(signal)))
        
        # Initialize SCSA once
        scsa = SCSA1D(gmma=req.gamma)
        
        # Process based on method
        if req.method == "filter_optimal":
            result = scsa.filter_with_optimal_h(noisy_abs)
        elif req.method == "denoise":
            reconstructed = scsa.denoise(noisy_abs)
            result = scsa.reconstruct(noisy_abs, h=req.h)
            result.reconstructed = reconstructed
        else:  # reconstruct
            result = scsa.reconstruct(noisy_abs, h=req.h)
        
        return {
            "x": x.tolist(),
            "signal": signal.tolist(),
            "noisy": noisy.tolist(),
            "reconstructed": result.reconstructed.tolist(),
            "metrics": result.metrics,
            "num_eigenvalues": result.num_eigenvalues
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SCSA processing error: {str(e)}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "SCSA API running", "version": "1.0.0"}


@app.get("/api/signal-types")
async def get_signal_types():
    """Return available signal types."""
    return {
        "signal_types": ["gaussian", "sech", "double_well", "chirp"],
        "methods": ["reconstruct", "denoise", "filter_optimal"]
    }
