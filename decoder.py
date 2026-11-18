"""Road-segment hidden states with classifier emissions and directed transitions."""
from itertools import groupby
import numpy as np
from scipy.special import logsumexp
from prepare import ROOT, load, project, candidates
from train_models import FEATURES

GAP_SECONDS = 30
MOTION_SCALE_M = 15


def decode(data, scores, arcs, times, motion_scale=MOTION_SCALE_M):
    n = len(times)
    baseline = np.full(n, -1, dtype=int)
    path = baseline.copy()
    offsets = np.searchsorted(data["observation_index"], np.arange(n + 1))
    history = []
    best = None
    resets = 0

    def finish():
        if history:
            state = int(best.argmax())
            for t, ids, parent in reversed(history):
                path[t] = ids[state]
                state = parent[state] if parent is not None else 0
            history.clear()

    for t in range(n):
        sl = slice(offsets[t], offsets[t+1])
        current = data[sl]
        ids = current["arc_id"]
        if not len(ids) or (t and times[t] - times[t-1] > GAP_SECONDS):
            finish()
        if not len(ids):
            continue
        emission = np.log(np.maximum(scores[sl], 1e-12))
        baseline[t] = ids[emission.argmax()]
        parent = None
        if history:
            prev = data[offsets[t-1]:offsets[t]]
            last = prev["arc_id"][:, None]
            same = last == ids
            linked = arcs[last, 1] == arcs[ids, 0]
            position = current["fraction"] * current["arc_length_m"]
            old = (prev["fraction"] * prev["arc_length_m"])[:, None]
            travel = np.where(same, abs(position-old), prev["arc_length_m"][:, None]-old+position)
            log_weight = -abs(travel-current["step_m"][0])/motion_scale
            log_weight = np.where(same | linked, log_weight, -np.inf)
            normalizer = logsumexp(log_weight, axis=1, keepdims=True)
            transitions = log_weight - np.where(np.isfinite(normalizer), normalizer, 0)
            joint = best[:, None] + transitions
            if np.isfinite(joint).any():
                parent = joint.argmax(axis=0)
                emission = emission + joint.max(axis=0)
            else:
                finish()
                resets += 1
        best = emission - emission.max()
        history.append((t, ids, parent))
    finish()
    return baseline, path, resets


def predict(trip, model):
    raw = load(trip, "track", float)
    nodes = load(trip, "nodes", float)
    arcs = load(trip, "arcs", int)
    origin = raw[:, :2].mean(axis=0)
    xy = project(nodes, origin)
    start = xy[arcs[:, 0]]
    delta = xy[arcs[:, 1]] - start
    points = project(raw[:, :2], origin)
    data = candidates(points, raw[:, 2], start, delta)
    scores = np.empty(0)
    if len(data):
        matrix = np.column_stack([data[key] for key in FEATURES]).astype("float32")
        scores = model.predict_proba(matrix)[:, 1]
    baseline, path, resets = decode(data, scores, arcs, raw[:, 2], float((ROOT / "experiments/motion_scale.txt").read_text()))
    return raw, nodes, arcs, points, start, delta, baseline, path, resets


def metrics(path, reference, arcs):
    from difflib import SequenceMatcher
    left, right = path[:-1], path[1:]
    valid = (left >= 0) & (right >= 0)
    continuous = valid & ((left == right) | (arcs[left, 1] == arcs[right, 0]))
    def sequence(values):
        return [int(k) for k, _ in groupby(values)]
    similarity = SequenceMatcher(None, sequence(reference), sequence(path), autojunk=False).ratio()
    return dict(observations=len(path), correct=int((path == reference).sum()),
                pairs=len(left), continuous=int(continuous.sum()), sequence_similarity=similarity)


