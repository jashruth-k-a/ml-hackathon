"""Train five fixed classifiers with trip-grouped CV and a separate student test."""
import csv
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments"
FEATURES = ["distance_m", "fraction", "arc_length_m", "heading_cosine", "elapsed_s", "step_m"]
SEED = 42


def load(partition):
    rows, targets, observations, groups, covered = [], [], [], [], []
    offset = 0
    for path in sorted((ROOT / "prepared" / partition).glob("*.features.npz")):
        name = path.name.split(".")[0]
        with np.load(path) as archive:
            data = archive["candidates"]
        with np.load(path.with_name(name + ".labels.npz")) as archive:
            reference = archive["reference_arc"]
            covered.append(archive["candidate_hit"])
        obs = data["observation_index"]
        targets.append(data["arc_id"] == reference[obs])
        rows.append(np.column_stack([data[key] for key in FEATURES]).astype("float32"))
        groups.append(np.full(len(reference), int(name)))
        observations.append(obs + offset)
        offset += len(reference)
    return tuple(map(np.concatenate, (rows, targets, observations, groups, covered)))


def evaluate(y, obs, scores, total, hits):
    # Stable ties retain candidate order (distance then arc ID).
    order = np.lexsort((-scores, obs))
    first = np.r_[True, np.diff(obs[order]) != 0]
    correct = int(y[order[first]].sum())
    binary = scores > 0.5
    return dict(top1=correct/total, conditional_top1=correct/hits,
                miss_rate=1-hits/total, macro_f1=f1_score(y, binary, average="macro"),
                precision=precision_score(y, binary, zero_division=0),
                recall=recall_score(y, binary, zero_division=0),
                log_loss=log_loss(y, np.clip(scores.astype(float), 1e-15, 1-1e-15)),
                confusion=confusion_matrix(y, binary).ravel().tolist())


def main():
    OUT.mkdir(exist_ok=True)
    train, test = load("train"), load("student_test")
    x, y, obs, groups, covered = train
    splits = list(GroupKFold(3).split(x, y, groups[obs]))
    folds = []
    runs = []
    for i, (fit, valid) in enumerate(splits, 1):
        ids = np.unique(groups[obs[valid]])
        folds.extend((i, f"{g:08d}") for g in ids)
        mask = np.isin(groups, ids)
        runs.append((f"cv{i}", fit, x[valid], y[valid], obs[valid], mask.sum(), covered[mask].sum()))
    with (OUT / "validation_trips.csv").open("w", newline="") as file:
        csv.writer(file).writerows([("fold", "trip_id"), *folds])
    tx, ty, tobs, tg, tc = test
    runs.append(("student_test", None, tx, ty, tobs, len(tg), tc.sum()))
    models = dict(logistic=make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)),
                  gaussian_nb=make_pipeline(StandardScaler(), GaussianNB()),
                  decision_tree=DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED),
                  random_forest=RandomForestClassifier(n_estimators=32, max_depth=14, min_samples_leaf=30,
                                                      n_jobs=4, random_state=SEED),
                  hist_boost=HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31,
                                                           early_stopping=False, random_state=SEED))
    with (OUT / "metrics.csv").open("w", newline="") as file:
        writer = None
        for name, model in models.items():
            for split, fit, vx, vy, vo, total, hits in runs:
                start = perf_counter()
                model.fit(x if fit is None else x[fit], y if fit is None else y[fit])
                fit_time = perf_counter() - start
                start = perf_counter()
                scores = model.predict_proba(vx)[:, 1]
                infer_time = perf_counter() - start
                results = evaluate(vy, vo, scores, total, hits)
                first = np.r_[True, np.diff(vo) != 0]
                row = dict(model=name, split=split, observations=total,
                           nearest_top1=vy[first].sum()/total, **results,
                           fit_seconds=fit_time, inference_seconds=infer_time)
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
                file.flush()
                print(name, split, round(results["top1"], 4), flush=True)
            joblib.dump(model, OUT / (name + ".joblib"), compress=3)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
