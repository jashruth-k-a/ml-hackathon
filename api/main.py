"""PathSnap API (Laptop 1).

First milestone: the UI on Laptop 2 lists trips, selects one, and the request
reaches this server and appears in the request log. The full prediction
pipeline (candidates -> classifier -> HMM/Viterbi -> scoring) is plugged into
`run_pipeline` in a later phase.

Run from the project root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables:
    PATHSNAP_DATA_DIR   folder containing one sub-folder per trip
                        (default: Student folder/train)
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA_DIR", ROOT / "Student folder" / "train")).resolve()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # UI runs on another laptop / origin
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    trip_id: str = Field(..., pattern=r"^\d{8}$", examples=["00000048"])


def list_trip_ids() -> list[str]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.track").is_file())


def load_track(trip_id: str) -> pd.DataFrame:
    path = DATA_DIR / trip_id / f"{trip_id}.track"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat", "t"])
    return df.dropna()


def run_pipeline(trip_id: str, track: pd.DataFrame) -> dict:
    """Placeholder: returns the observed GPS trace only.

    Ground-truth routes are never read here. They will only be used server-side
    for scoring once the prediction pipeline is connected.
    """
    return {
        "status": "connected",
        "message": "Request received. Prediction pipeline not connected yet.",
        "predicted_route": None,
        "metrics": None,
    }


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    client = request.client.host if request.client else "?"
    log.info("%s %s from %s -> %d (%.1f ms)", request.method, request.url.path, client, response.status_code, ms)
    return response


@app.on_event("startup")
def startup() -> None:
    trips = list_trip_ids()
    log.info("Data dir: %s", DATA_DIR)
    log.info("Trips available: %d", len(trips))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "trips": len(list_trip_ids())}


@app.get("/trips")
def trips() -> dict:
    return {"trips": list_trip_ids()}


@app.post("/predict")
def predict(req: PredictRequest, request: Request) -> dict:
    t0 = time.perf_counter()
    client = request.client.host if request.client else "?"
    log.info("PREDICT trip=%s requested by %s", req.trip_id, client)

    track = load_track(req.trip_id)
    result = run_pipeline(req.trip_id, track)
    elapsed = (time.perf_counter() - t0) * 1000

    log.info("PREDICT trip=%s done: %d GPS points, %.1f ms", req.trip_id, len(track), elapsed)
    return {
        "trip_id": req.trip_id,
        "n_points": int(len(track)),
        "duration_s": float(track["t"].iloc[-1] - track["t"].iloc[0]) if len(track) else 0.0,
        "gps": track[["lat", "lon"]].round(6).values.tolist(),
        "processing_ms": round(elapsed, 1),
        **result,
    }
