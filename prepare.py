"""Build candidate features independently of development reference routes."""
from pathlib import Path
import argparse
import csv
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent
RADIUS = 75.0
LIMIT = 12


def load(trip, suffix, dtype):
    return np.loadtxt(trip / f"{trip.name}.{suffix}", dtype=dtype, ndmin=2)


def project(coords, origin):
    scale = np.array([np.cos(np.radians(origin[1])), 1]) * (6371008.8 * np.pi / 180)
    return (coords - origin) * scale


def geometry(point, start, delta):
    offset = point - start
    length2 = (delta * delta).sum(axis=-1)
    fraction = np.clip((offset * delta).sum(axis=-1) / np.maximum(length2, 1e-12), 0, 1)
    residual = offset - fraction[..., None] * delta
    return np.linalg.norm(residual, axis=-1), fraction


def candidates(points, times, start, delta):
    length = np.linalg.norm(delta, axis=1)
    pieces = np.maximum(1, np.ceil(length / 20).astype(int))
    ids = np.repeat(np.arange(len(start)), pieces)
    first = np.cumsum(pieces) - pieces
    fraction = (np.arange(len(ids)) - first[ids] + 0.5) / pieces[ids]
    tree = cKDTree(start[ids] + fraction[:, None] * delta[ids])
    movement = np.diff(points, axis=0, prepend=points[:1])
    step = np.linalg.norm(movement, axis=1)
    dt = np.diff(times, prepend=times[:1])
    chunks = []
    for base in range(0, len(points), 256):
        hits = tree.query_ball_point(points[base:base + 256], RADIUS + 10.00001)
        sizes = np.array([len(hit) for hit in hits])
        if not sizes.sum():
            continue
        obs = np.repeat(np.arange(base, base + len(hits)), sizes)
        arc = ids[np.concatenate([hit for hit in hits if len(hit)])]
        obs, arc = np.divmod(np.unique(obs * len(start) + arc), len(start))
        dist, pos = geometry(points[obs], start[arc], delta[arc])
        order = np.lexsort((arc, dist, obs))
        order = order[dist[order] <= RADIUS]
        if not len(order):
            continue
        group = obs[order]
        starts = np.r_[True, group[1:] != group[:-1]]
        rank = np.arange(len(group)) - np.maximum.accumulate(np.where(starts, np.arange(len(group)), 0))
        ix = order[rank < LIMIT]
        o, a = obs[ix], arc[ix]
        norm = step[o] * length[a]
        cosine = np.divide((movement[o] * delta[a]).sum(axis=1), norm,
                           out=np.zeros(len(o)), where=norm > 0)
        chunks.append(np.rec.fromarrays([o, a, dist[ix], pos[ix], length[a], cosine,
                                        dt[o], step[o]], dtype=FEATURES))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=FEATURES)


FEATURES = np.dtype([("observation_index","i4"), ("arc_id","i4"), ("distance_m","f4"),
                     ("fraction","f4"), ("arc_length_m","f4"), ("heading_cosine","f4"),
                     ("elapsed_s","f8"), ("step_m","f4")])


def align(points, start, delta, route):
    # Minimum total point-to-segment distance under chronological route order.
    s, d = start[route], delta[route]
    parents = np.empty((len(points), len(route)), dtype=np.int32)
    previous = np.zeros(len(route))
    indices = np.arange(len(route))
    for base in range(0, len(points), 64):
        distances, _ = geometry(points[base:base+64, None], s, d)
        for offset, distances_i in enumerate(distances):
            best = np.minimum.accumulate(previous)
            parents[base + offset] = np.maximum.accumulate(np.where(previous == best, indices, 0))
            previous = best + distances_i
    path = np.empty(len(points), dtype=int)
    state = previous.argmin()
    for i in reversed(range(len(points))):
        path[i] = state
        state = parents[i, state]
    ref = route[path]
    residual, _ = geometry(points, start[ref], delta[ref])
    ambiguous = np.zeros(len(points), dtype=bool)
    for base in range(0, len(points), 64):
        sl = slice(base, base+64)
        distances, _ = geometry(points[sl, None], s, d)
        ambiguous[sl] = ((route != ref[sl,None]) & (distances <= residual[sl,None] + 3)).any(axis=1)
    return ref, residual, ambiguous


def process(trip, dest):
    raw = load(trip, "track", float)
    nodes = load(trip, "nodes", float)
    arcs = load(trip, "arcs", int)
    assert np.isfinite(raw).all() and np.isfinite(nodes).all()
    assert arcs.min() >= 0 and arcs.max() < len(nodes)
    origin = raw[:, :2].mean(axis=0)
    points, xy = project(raw[:, :2], origin), project(nodes, origin)
    start = xy[arcs[:,0]]
    delta = xy[arcs[:,1]] - start
    features = candidates(points, raw[:,2], start, delta)
    np.savez_compressed(dest / f"{trip.name}.features.npz", candidates=features)
    route = load(trip, "route", int).ravel()
    assert route.min() >= 0 and route.max() < len(arcs)
    ref, residual, ambiguous = align(points, start, delta, route)
    obs, arc = features["observation_index"], features["arc_id"]
    present = np.zeros(len(raw), dtype=bool)
    present[obs[arc == ref[obs]]] = True
    np.savez_compressed(dest / f"{trip.name}.labels.npz", reference_arc=ref,
                        residual_m=residual, ambiguous=ambiguous, candidate_hit=present)
    graph = coo_matrix((np.ones(len(arcs), dtype=bool), arcs.T), shape=(len(nodes), len(nodes))).tocsr()
    components, groups = connected_components(graph, directed=False)
    isolated = (np.bincount(arcs.ravel(), minlength=len(nodes)) == 0).sum()
    return [len(raw), len(features), len(raw)-len(np.unique(obs)), (~present).sum(),
            ambiguous.sum(), (residual > RADIUS).sum(), components, isolated,
            np.bincount(groups).max(), *raw[:,:2].min(axis=0), *raw[:,:2].max(axis=0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip", help="Optional development trip ID")
    args = parser.parse_args()
    rows = []
    for partition in ("train", "student_test"):
        dest = ROOT / "prepared" / partition
        dest.mkdir(exist_ok=True, parents=True)
        for trip in sorted((ROOT / "Student folder" / partition).iterdir()):
            if trip.is_dir() and (args.trip is None or args.trip == trip.name):
                rows.append([partition, trip.name, *process(trip, dest)])
    if not rows:
        parser.error("No matching development trips")
    output = ROOT / ("preparation_summary.csv" if not args.trip else f"prepared/{args.trip}.csv")
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow("partition trip_id observations candidates empty misses ambiguous distant weak_components isolated largest_component min_lon min_lat max_lon max_lat".split())
        writer.writerows(rows)


if __name__ == "__main__":
    main()

