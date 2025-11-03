from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from pyscsa import SCSA1D

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SCSARequest(BaseModel):
    gamma: float = 0.5
    h: float = 1.0
    noise: float = 0.1
    signal_type: str = "gaussian"
    method: str = "reconstruct"  # reconstruct, denoise, filter_optimal
    
def generate_signal(signal_type: str, x: np.ndarray):
    if signal_type == "gaussian":
        return np.exp(-x**2)
    elif signal_type == "sech":
        return 1 / np.cosh(x)
    elif signal_type == "double_well":
        return -50 * (1/np.cosh(x-3))**2 - 50 * (1/np.cosh(x+3))**2
    elif signal_type == "chirp":
        return np.sin(x**2)
    return np.exp(-x**2)
        return np.sin(x**2)
    return np.exp(-x**2)

@app.post("/api/scsa")
async def run_scsa(req: SCSARequest):
    x = np.linspace(-10, 10, 200)
    signal = generate_signal(req.signal_type, x)
    noisy = signal + req.noise * np.random.randn(len(signal))
    
    noisy = signal + req.noise * np.random.randn(len(signal))

    scsa = SCSA1D(gmma=req.gamma)

    if req.method == "filter_optimal":
        result = scsa.filter_with_c_scsa(np.abs(noisy))
    elif req.method == "denoise":
        reconstructed = scsa.denoise(np.abs(noisy))
        result = scsa.reconstruct(np.abs(noisy), h=req.h)
        result.reconstructed = reconstructed
    else:
        result = scsa.reconstruct(np.abs(noisy), h=req.h)

    return {
        "x": x.tolist(),
        "signal": signal.tolist(),
        "noisy": noisy.tolist(),
        "reconstructed": result.reconstructed.tolist(),
        "metrics": result.metrics,
        "num_eigenvalues": result.num_eigenvalues
    }

@app.get("/")
async def root():
    return {"status": "SCSA API running"}
