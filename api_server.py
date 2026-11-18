"""Laptop 1: prediction first, then isolated reference-based scoring."""
import os
from functools import lru_cache
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from decoder import metrics, predict
from prepare import ROOT, align, load

app = FastAPI(title="PathSnap inference")
PARTITIONS = {"student": ROOT / "Student folder/student_test", "evaluation": ROOT / "evaluator_test"}
DATA = PARTITIONS[os.environ.get("PATHSNAP_MODE", "student")]
MODEL = ROOT / "experiments" / "selected_model.txt"


@lru_cache(maxsize=1)
def classifier():
    return joblib.load(MODEL.parent / MODEL.read_text().strip())


@app.get("/trips")
def trips():
    return sorted(p.name for p in DATA.iterdir() if p.is_dir() and len(p.name)==8 and p.name.isdigit())


@app.post("/trips/{trip_id}/predict")
def run(trip_id: str):
    if trip_id not in trips():
        raise HTTPException(404, "Unknown trip")
    trip = DATA / trip_id
    raw, nodes, arcs, points, start, delta, _, path, resets = predict(trip, classifier())
    # Only the scorer reads the route. Reference arrays never enter the response.
    reference, _, _ = align(points, start, delta, load(trip, "route", int).ravel())
    m = metrics(path, reference, arcs)
    ids = path[path >= 0]
    return dict(trip_id=trip_id, gps=raw[:, :2].tolist(),
                segments=nodes[arcs[np.unique(ids)]].tolist(), predicted_arc_ids=path.tolist(),
                observations=len(path), matched=int(len(ids)), restarts=resets,
                accuracy=m["correct"]/len(path), continuity=m["continuous"]/max(1,m["pairs"]))


