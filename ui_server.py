"""Laptop 2: serve the interface locally and forward requests to Laptop 1."""
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pathlib import Path

app = FastAPI(title="PathSnap interface")
BACKEND = os.environ.get("PATHSNAP_API","http://127.0.0.1:8000").rstrip("/")


def forward(path, method):
    try:
        with urlopen(Request(BACKEND+path, method=method), timeout=180) as r:
            return Response(content=r.read(), media_type="application/json")
    except HTTPError as exc:
        raise HTTPException(exc.code, "Prediction service rejected the request") from exc
    except (URLError, TimeoutError) as exc:
        raise HTTPException(502, "Cannot reach the prediction laptop") from exc


@app.get("/")
def home():
    return FileResponse(Path(__file__).with_name("interface.html"))


@app.get("/trips")
def trips():
    return forward("/trips","GET")


@app.post("/trips/{trip_id}/predict")
def predict(trip_id: str):
    if len(trip_id)!=8 or not trip_id.isascii() or not trip_id.isdecimal():
        raise HTTPException(404,"Unknown trip")
    return forward("/trips/"+trip_id+"/predict","POST")
