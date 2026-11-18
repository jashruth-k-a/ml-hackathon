"""Compare full-fit training macro F1 with held-out trip CV for each initial model."""
import csv
from statistics import mean
import joblib
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from train_models import OUT, load


def main():
    x, y, *_ = load("train")
    with (OUT / "metrics.csv").open() as source:
        runs = list(csv.DictReader(source))
    with (OUT / "fit_diagnostics.csv").open("w", newline="") as dest:
        writer = csv.writer(dest)
        writer.writerow(["model", "training_macro_f1", "cv_macro_f1", "difference"])
        for row in runs:
            if row["split"] != "student_test":
                continue
            name = row["model"]
            model = joblib.load(OUT / (name + ".joblib"))
            training = f1_score(y, model.predict(x), average="macro")
            validation = mean(float(r["macro_f1"]) for r in runs
                              if r["model"] == name and r["split"].startswith("cv"))
            writer.writerow([name, training, validation, training-validation])


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
