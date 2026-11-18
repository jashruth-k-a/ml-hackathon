"""Validate prepared development artifacts and compare sampled candidates to exhaustive search."""
import csv
import numpy as np
from prepare import ROOT, LIMIT, RADIUS, geometry, load, project


def main():
    with (ROOT / "development_inventory.csv").open() as f:
        inventory = list(csv.DictReader(f))
    checked = 0
    for row in inventory:
        partition, name = row["partition"], row["trip_id"]
        trip = ROOT / "Student folder" / partition / name
        dest = ROOT / "prepared" / partition
        with np.load(dest / f"{name}.features.npz") as archive:
            features = archive["candidates"]
        with np.load(dest / f"{name}.labels.npz") as labels:
            ref = labels["reference_arc"]
            hits = labels["candidate_hit"]
        obs, arc = features["observation_index"], features["arc_id"]
        raw = load(trip, "track", float)
        arcs = load(trip, "arcs", int)
        nodes = load(trip, "nodes", float)
        origin = raw[:,:2].mean(axis=0)
        xy = project(nodes, origin)
        start = xy[arcs[:,0]]
        delta = xy[arcs[:,1]] - start
        assert len(ref) == len(raw) == int(row["track"])
        assert np.isin(ref, load(trip, "route", int)).all()
        assert np.all((obs >= 0) & (obs < len(ref)))
        assert np.all((arc >= 0) & (arc < len(arcs)))
        assert np.all(np.diff(obs) >= 0)
        assert max(np.bincount(obs, minlength=len(ref))) <= LIMIT
        assert np.unique(obs.astype("int64") * len(arcs) + arc).size == len(obs)
        assert all(np.isfinite(features[field]).all() for field in features.dtype.names)
        assert np.array_equal(np.flatnonzero(hits), np.unique(obs[arc == ref[obs]]))
        for i in (0, len(raw)//2, len(raw)-1):
            distance, _ = geometry(project(raw[i,:2], origin), start, delta)
            valid = np.flatnonzero(distance <= RADIUS)
            expected = valid[np.lexsort((valid, distance[valid]))[:LIMIT]]
            assert np.array_equal(arc[obs == i], expected), (name, i)
            checked += 1
    print(f"Verified {len(inventory)} trips and {checked} exhaustive candidate comparisons.")


if __name__ == "__main__":
    main()
