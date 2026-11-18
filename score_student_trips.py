"""Summarise independent student-test predictions from each saved classifier."""
import csv
import joblib
import numpy as np
from train_models import FEATURES, OUT, ROOT
from threadpoolctl import threadpool_limits


def main():
    with (OUT / "student_trips.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "trip_id", "observations", "correct", "continuous_pairs", "pairs"])
        for artifact in sorted(OUT.glob("*.joblib")):
            model = joblib.load(artifact)
            for path in sorted((ROOT / "prepared/student_test").glob("*.features.npz")):
                name = path.name.split(".")[0]
                with np.load(path) as archive:
                    data = archive["candidates"]
                with np.load(path.with_name(name + ".labels.npz")) as archive:
                    ref = archive["reference_arc"]
                arcs = np.loadtxt(ROOT / "Student folder/student_test" / name / (name + ".arcs"), dtype=int)
                features = np.column_stack([data[k] for k in FEATURES]).astype("float32")
                scores = model.predict_proba(features)[:, 1]
                obs = data["observation_index"]
                order = np.lexsort((-scores, obs))
                winners = order[np.r_[True, np.diff(obs[order]) != 0]]
                pred = np.full(len(ref), -1)
                pred[obs[winners]] = data["arc_id"][winners]
                a, b = pred[:-1], pred[1:]
                valid = (a >= 0) & (b >= 0)
                continuous = valid & ((a == b) | (arcs[a, 1] == arcs[b, 0]))
                writer.writerow([artifact.stem, name, len(ref), sum(pred == ref),
                                 continuous.sum(), len(a)])


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
